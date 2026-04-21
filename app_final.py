import cv2, numpy as np, tensorflow as tf, mediapipe as mp, threading

try:
    import pyttsx3
    TTS=True
except:
    TTS=False

def speak(text):
    if not text or not TTS:
        return
    def run(t):
        try:
            e=pyttsx3.init()
            e.say(t)
            e.runAndWait()
        except:
            pass
    threading.Thread(target=run,args=(text,),daemon=True).start()

class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def build(self, input_shape):
        self.W=self.add_weight(shape=(input_shape[-1],1),initializer='random_normal',trainable=True)
        self.b=self.add_weight(shape=(input_shape[1],1),initializer='zeros',trainable=True)
    def call(self,x):
        e=tf.matmul(x,self.W)+self.b
        e=tf.nn.tanh(e)
        a=tf.nn.softmax(e,axis=1)
        return tf.reduce_sum(x*a,axis=1)

coarse_model=tf.keras.models.load_model("cv_fusion/coarse_model/coarse_model.h5",
    custom_objects={"AttentionLayer":AttentionLayer},compile=False)

coarse_labels=np.load("cv_fusion/label_maps/coarse_labels.npy",allow_pickle=True).item()
coarse_classes={v:k for k,v in coarse_labels.items()}

fine_models={}
fine_labels={}

for g in coarse_classes.values():
    fine_models[g]=tf.keras.models.load_model(f"cv_fusion/fine_models/{g}/model.h5",
        custom_objects={"AttentionLayer":AttentionLayer},compile=False)
    lab=np.load(f"cv_fusion/label_maps/{g}_labels.npy",allow_pickle=True).item()
    fine_labels[g]={v:k for k,v in lab.items()}

doodle_model = tf.keras.models.load_model("cv_doodle/doodle_cnn.h5", compile=False)
doodle_labels = np.load("cv_doodle/labels.npy", allow_pickle=True)

mp_hands=mp.solutions.hands
hands=mp_hands.Hands(static_image_mode=False,max_num_hands=2,min_detection_confidence=0.2,min_tracking_confidence=0.5)

cap=cv2.VideoCapture(0)

H,W=480,640
canvas=np.zeros((H,W),np.uint8)

sentence=""
word=""

undo_stack=[]

cooldown=0
action_cd=0
switch_cd=0

fist_count=0
two_count=0
thumb_count=0

THRESH=12

prev_x, prev_y = None, None
mode = "WORD"

left_prev_x = None
left_swipe_start = None
left_swipe_frames = 0

LEFT_SWIPE_THRESHOLD = 120
LEFT_SWIPE_FRAMES = 6

def preprocess(img):
    coords=cv2.findNonZero(img)
    if coords is None:
        return None
    x,y,w,h=cv2.boundingRect(coords)
    roi=img[y:y+h,x:x+w]
    roi=cv2.copyMakeBorder(roi,20,20,20,20,cv2.BORDER_CONSTANT,value=0)
    roi=cv2.resize(roi,(64,64))
    roi=cv2.threshold(roi,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    roi=cv2.GaussianBlur(roi,(3,3),0)
    roi=roi.astype("float32")/255.0
    return roi.reshape(1,64,64,1)

def recognize_word(img):
    roi=preprocess(img)
    if roi is None:
        return None,0.0
    cp=coarse_model.predict(roi,verbose=0)[0]
    g=coarse_classes[int(np.argmax(cp))]
    fp=fine_models[g].predict(roi,verbose=0)[0]
    idx=int(np.argmax(fp))
    return fine_labels[g][idx],float(np.max(fp))

def recognize_doodle(img):
    roi=preprocess(img)
    if roi is None:
        return None,0.0
    pred=doodle_model.predict(roi,verbose=0)[0]
    idx=int(np.argmax(pred))
    return doodle_labels[idx],float(np.max(pred))

def index_only(lm):
    return lm.landmark[8].y<lm.landmark[6].y and lm.landmark[12].y>lm.landmark[10].y and lm.landmark[16].y>lm.landmark[14].y and lm.landmark[20].y>lm.landmark[18].y

def open_hand(lm):
    return lm.landmark[8].y<lm.landmark[6].y and lm.landmark[12].y<lm.landmark[10].y and lm.landmark[16].y<lm.landmark[14].y and lm.landmark[20].y<lm.landmark[18].y

def fist(lm):
    return lm.landmark[8].y>lm.landmark[6].y and lm.landmark[12].y>lm.landmark[10].y and lm.landmark[16].y>lm.landmark[14].y and lm.landmark[20].y>lm.landmark[18].y

def two_finger(lm):
    return lm.landmark[8].y<lm.landmark[6].y and lm.landmark[12].y<lm.landmark[10].y and lm.landmark[16].y>lm.landmark[14].y and lm.landmark[20].y>lm.landmark[18].y

def thumb_up_only(lm):
    return lm.landmark[4].y<lm.landmark[3].y and lm.landmark[8].y>lm.landmark[6].y and lm.landmark[12].y>lm.landmark[10].y and lm.landmark[16].y>lm.landmark[14].y and lm.landmark[20].y>lm.landmark[18].y

def open_hand_all(lm):
    return lm.landmark[8].y<lm.landmark[6].y and lm.landmark[12].y<lm.landmark[10].y and lm.landmark[16].y<lm.landmark[14].y and lm.landmark[20].y<lm.landmark[18].y

while True:
    ret,frame=cap.read()
    if not ret:
        break

    frame=cv2.flip(frame,1)
    frame=cv2.resize(frame,(W,H))

    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    res=hands.process(rgb)

    right_hand=None
    left_hand=None

    if res.multi_hand_landmarks and res.multi_handedness:
        for i,lm in enumerate(res.multi_hand_landmarks):
            label=res.multi_handedness[i].classification[0].label
            if label=="Right": right_hand=lm
            if label=="Left": left_hand=lm

    if left_hand:
        lm=left_hand
        lx=int(lm.landmark[8].x*W)

        if thumb_up_only(lm):
            thumb_count+=1
            if thumb_count>THRESH and switch_cd==0:
                mode="DOODLE" if mode=="WORD" else "WORD"
                switch_cd=25
                thumb_count=0
        else:
            thumb_count=0

        if open_hand_all(lm):
            if left_prev_x is not None:
                dx = lx - left_prev_x

                if dx < -15:
                    if left_swipe_start is None:
                        left_swipe_start = left_prev_x
                    left_swipe_frames += 1

                elif dx > 15:
                    if left_swipe_start is None:
                        left_swipe_start = left_prev_x
                    left_swipe_frames += 1

                else:
                    left_swipe_start = None
                    left_swipe_frames = 0

                if left_swipe_start is not None:
                    dist = lx - left_swipe_start

                    if dist < -LEFT_SWIPE_THRESHOLD and left_swipe_frames < LEFT_SWIPE_FRAMES:
                        if len(sentence)>0:
                            words = sentence.strip().split(" ")
                            last = words[-1]
                            undo_stack.append(last)
                            sentence = " ".join(words[:-1]) + " "
                        left_swipe_start = None
                        left_swipe_frames = 0

                    elif dist > LEFT_SWIPE_THRESHOLD and left_swipe_frames < LEFT_SWIPE_FRAMES:
                        if len(undo_stack)>0:
                            word = undo_stack.pop()
                            sentence += word + " "
                        left_swipe_start = None
                        left_swipe_frames = 0

            left_prev_x = lx
        else:
            left_prev_x = None
            left_swipe_start = None
            left_swipe_frames = 0

    if right_hand:
        lm=right_hand
        cx=int(lm.landmark[8].x*W)
        cy=int(lm.landmark[8].y*H)

        if index_only(lm):
            if prev_x is not None:
                cv2.line(canvas,(prev_x,prev_y),(cx,cy),255,5)
            prev_x,prev_y=cx,cy

        elif open_hand(lm):
            if cooldown==0:
                if mode=="WORD":
                    label,conf=recognize_word(canvas)
                    if label and conf>0.6:
                        word+=label.upper()
                        sentence+=word+" "
                        speak(word)
                        word=""
                else:
                    label,conf=recognize_doodle(canvas)
                    if label and conf>0.6:
                        label=str(label).upper()
                        sentence+=label+" "
                        speak(label)

                canvas.fill(0)
                cooldown=10

            prev_x,prev_y=None,None

        elif fist(lm):
            fist_count+=1
            if fist_count>THRESH and action_cd==0:
                canvas.fill(0)
                sentence=""
                word=""
                undo_stack.clear()
                action_cd=25
                fist_count=0

            prev_x,prev_y=None,None

        elif two_finger(lm):
            two_count+=1
            if two_count>THRESH and action_cd==0:
                sentence=sentence[:-1]
                action_cd=20
                two_count=0

            prev_x,prev_y=None,None

        else:
            prev_x,prev_y=None,None

    if cooldown>0: cooldown-=1
    if action_cd>0: action_cd-=1
    if switch_cd>0: switch_cd-=1

    cv2.putText(frame,f"Mode: {mode}",(10,40),
        cv2.FONT_HERSHEY_SIMPLEX,1,
        (0,255,0) if mode=="WORD" else (255,0,0),2)

    cv2.putText(frame,sentence,(10,80),
        cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

    overlay=cv2.addWeighted(frame,1.0,cv2.cvtColor(canvas,cv2.COLOR_GRAY2BGR),0.5,0)
    cv2.imshow("Air Writing + Doodle",overlay)

    if cv2.waitKey(1)&0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
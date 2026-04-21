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

coarse_model=tf.keras.models.load_model("coarse_model/coarse_model.h5",custom_objects={"AttentionLayer":AttentionLayer},compile=False)

coarse_labels=np.load("label_maps/coarse_labels.npy",allow_pickle=True).item()
coarse_classes={v:k for k,v in coarse_labels.items()}

fine_models={}
fine_labels={}

for g in coarse_classes.values():
    fine_models[g]=tf.keras.models.load_model(f"fine_models/{g}/model.h5",custom_objects={"AttentionLayer":AttentionLayer},compile=False)
    lab=np.load(f"label_maps/{g}_labels.npy",allow_pickle=True).item()
    fine_labels[g]={v:k for k,v in lab.items()}

mp_hands=mp.solutions.hands
hands=mp_hands.Hands(static_image_mode=False,max_num_hands=1,min_detection_confidence=0.5,min_tracking_confidence=0.5)

cap=cv2.VideoCapture(0)

H,W=480,640
canvas=np.zeros((H,W),np.uint8)

drawing=[]
is_drawing=False
sentence=""
word=""
cooldown=0
action_cd=0

fist_count=0
two_count=0
THRESH=12

prev_x, prev_y = 0, 0

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

def recognize(img):
    roi=preprocess(img)
    if roi is None:
        return None,0.0
    cp=coarse_model.predict(roi,verbose=0)[0]
    g=coarse_classes[int(np.argmax(cp))]
    fp=fine_models[g].predict(roi,verbose=0)[0]
    idx=int(np.argmax(fp))
    conf=float(np.max(fp))
    return fine_labels[g][idx],conf

def index_only(lm):
    return lm.landmark[8].y<lm.landmark[6].y and lm.landmark[12].y>lm.landmark[10].y and lm.landmark[16].y>lm.landmark[14].y and lm.landmark[20].y>lm.landmark[18].y

def open_hand(lm):
    return lm.landmark[8].y<lm.landmark[6].y and lm.landmark[12].y<lm.landmark[10].y and lm.landmark[16].y<lm.landmark[14].y and lm.landmark[20].y<lm.landmark[18].y

def fist(lm):
    return lm.landmark[8].y>lm.landmark[6].y and lm.landmark[12].y>lm.landmark[10].y and lm.landmark[16].y>lm.landmark[14].y and lm.landmark[20].y>lm.landmark[18].y

def two_finger(lm):
    index_up = lm.landmark[8].y < lm.landmark[6].y
    middle_up = lm.landmark[12].y < lm.landmark[10].y
    ring_down = lm.landmark[16].y > lm.landmark[14].y
    pinky_down = lm.landmark[20].y > lm.landmark[18].y
    return index_up and middle_up and ring_down and pinky_down

while True:
    ret,frame=cap.read()
    if not ret:
        break
    frame=cv2.flip(frame,1)
    frame=cv2.resize(frame,(W,H))
    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    res=hands.process(rgb)
    if res.multi_hand_landmarks:
        lm=res.multi_hand_landmarks[0]
        cx,cy=int(lm.landmark[8].x*W),int(lm.landmark[8].y*H)
        movement = abs(cx - prev_x) + abs(cy - prev_y)
        prev_x, prev_y = cx, cy

        if index_only(lm):
            fist_count=0
            two_count=0
            if not is_drawing:
                drawing=[(cx,cy)]
                is_drawing=True
            else:
                if len(drawing)>0:
                    cv2.line(canvas,drawing[-1],(cx,cy),255,5)
                drawing.append((cx,cy))

        elif open_hand(lm):
            fist_count=0
            two_count=0
            if len(drawing)>10 and cooldown==0:
                label,conf=recognize(canvas)
                if label and conf>0.6:
                    word+=label.upper()
                    sentence+=word+" "
                    speak(word)
                    word=""
                canvas.fill(0)
                drawing.clear()
                cooldown=10
            is_drawing=False

        elif fist(lm):
            fist_count+=1
            two_count=0

            if fist_count>THRESH and action_cd==0:
                canvas.fill(0)
                drawing.clear()
                sentence=""
                word=""
                action_cd=25
                fist_count=0

        elif two_finger(lm) and movement < 5:
            two_count+=1
            fist_count=0

            if two_count>THRESH and action_cd==0:
                if len(sentence)>0:
                    sentence=sentence[:-1]
                action_cd=20
                two_count=0

        else:
            fist_count=0
            two_count=0
            is_drawing=False

    if cooldown>0:
        cooldown-=1
    if action_cd>0:
        action_cd-=1

    cv2.putText(frame,sentence,(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
    overlay=cv2.addWeighted(frame,1.0,cv2.cvtColor(canvas,cv2.COLOR_GRAY2BGR),0.5,0)
    cv2.imshow("Air Writing",overlay)
    if cv2.waitKey(1)&0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
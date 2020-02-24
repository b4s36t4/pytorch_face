import numpy as np
import cv2
from facenet_pytorch import MTCNN
import torch
from PIL import Image,ImageDraw
cap = cv2.VideoCapture(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Running %s"%device)

mtcnn = MTCNN(keep_all=True,device=device)

i = 1
frames_tracked = []
while True:
    ret,fram = cap.read()
    gray = cv2.cvtColor(fram,cv2.COLOR_BGR2RGB)

    frame = Image.fromarray(gray)
    #print('\rTracking frame: {}'.format(i + 1), end='')
    boxes, _ = mtcnn.detect(frame)
    
    # Draw faces
    frame_draw = frame.copy()
    draw = ImageDraw.Draw(frame_draw)
    for box in boxes:
        draw.rectangle(box.tolist(), outline=(255, 0, 0), width=6)
    
    # Add to frame list
    abc = frame_draw.resize((640, 360))
    frames_tracked.append(abc)
    x = np.array(abc)
    cv2.imshow("frame",x)
    print("Showing.. %s"%i)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    i = i+1

cap.release()
cv2.destroyAllWindows()

dim = frames_tracked[0].size
fourcc = cv2.VideoWriter_fourcc(*"FMP4")
v_t = cv2.VideoWriter("tracked.mp4",fourcc,10.0,dim)
for frame in frames_tracked:
    v_t.write(cv2.cvtColor(np.array(frame),cv2.COLOR_RGB2BGR))

v_t.release()

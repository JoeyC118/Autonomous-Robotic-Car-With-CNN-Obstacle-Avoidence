from sunfounder_controller import SunFounderController
from picarx import Picarx
from robot_hat import utils
from vilib import Vilib
import cv2
from time import sleep
import threading
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
import torchvision.transforms as transforms 
from PIL import Image
from collections import deque
import statistics

recent_preds = deque(maxlen=3)

# reset robot_hat
utils.reset_mcu()
sleep(0.2)

# init SunFounder Controller class
sc = SunFounderController()
sc.set_name('Picarx-001')
sc.set_type('Picarx')
sc.start()

# init picarx
px = Picarx()
speed = 0

#pytorch CNN network
class DrivingNet(nn.Module):
    def __init__(self, conv1_out, conv2_out, conv3_out, fc1_out, fc2_out):
        super().__init__()

        #defining model architecture
        self.conv1 = nn.Conv2d(3, conv1_out, kernel_size = 5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(conv1_out,conv2_out,kernel_size = 5)
        self.conv3 = nn.Conv2d(conv2_out, conv3_out, kernel_size=3)


        self.fc1 = nn.Linear(conv3_out*1*1, fc1_out)
        self.fc2 = nn.Linear(fc1_out,fc2_out)
        self.fc3 = nn.Linear(fc2_out,2)

        #dropout layer for increased performance 
        self.dropout = nn.Dropout(0.2)

        #send frame through model
    def forward(self,x): 
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x,1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x 


#loading model weights before autonomous driving
print("Starting Model")
model = DrivingNet(16,32,64,128,64)
model.load_state_dict(torch.load('autonomous_model_weights.pth'))
model.eval()


def main():
    global speed
    ip = utils.get_ip()
    video_url = f"http://{ip}:9000/mjpg"
    print('ip : %s'%ip)
    sc.set('video',video_url)

    Vilib.camera_start(vflip=False,hflip=False)
    Vilib.display(local=False, web=True)
    transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5] *3)
    ])


    while True:    
        #collecting the frame from the robot webcam
        cap = cv2.VideoCapture(f"http://{ip}:9000/mjpg")
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            print("no frame")
            continue

        #converting image so it can be sent through neural net
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame_tensor = transform(frame)
        frame_tensor = frame_tensor.unsqueeze(0)

        #sending the data through and adding it to queue
        with torch.no_grad():
            output = model(frame_tensor)
            predicted = torch.argmax(output, dim = 1).item()
            recent_preds.append(predicted)

            #taking an average of the recent predictions in order to make better decision
            try:
                smoothed_pred = statistics.mode(recent_preds)
            except statistics.StatisticsError:
                smoothed_pred = predicted


        #turn left
        if smoothed_pred == 0:
            print("predicting left")
            prev_action = "left"
            px.set_dir_servo_angle(-30)
            px.forward(30) 
        #turn right
        else:
            print("predicting right")
            prev_action = "right"
            px.set_dir_servo_angle(30)
            px.forward(30)
        sleep(0.6)
        



if __name__ == "__main__":
    try:
        main()
    finally:
        print("stop and exit")
        px.stop()
        Vilib.camera_close()
    

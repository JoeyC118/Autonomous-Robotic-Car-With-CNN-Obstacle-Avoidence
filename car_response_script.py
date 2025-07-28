from sunfounder_controller import SunFounderController
from picarx import Picarx
from robot_hat import utils
from vilib import Vilib
import os
import cv2
from time import sleep
import threading

#robot car initalization
utils.reset_mcu()
sleep(0.2)
sc = SunFounderController()
sc.set_name('Picarx-001')
sc.set_type('Picarx')
sc.start()
px = Picarx()
speed = 0


#used to capture frames when collecting data
def save_frames(frame_id, label):
   cap = cv2.VideoCapture("http://127.0.0.1:9000/mjpg")
   ret, frame = cap.read()
   cap.release()
   if not ret or frame is None:
      print("❌ Failed to capture frame — skipping save.")
   else:
      filename = f"frame_{frame_id:04d}.jpg"
      cv2.imwrite(f"data/{filename}", frame)
      with open("labels.csv", "a") as data_file:
         data_file.write(f"{filename},{label}\n")
      print(f"Saved Frame {filename} labeled as '{label}'")
    
def main():
    global speed
    ip = utils.get_ip()
    print('ip : %s'%ip)
    sc.set('video','http://'+ip+':9000/mjpg')
    Vilib.camera_start(vflip=False,hflip=False)
    Vilib.display(local=False, web=True)
    frame_id = len(os.listdir("data")) + 1
    prev_action = ""

    while True:
        
        #code that handles commands
        sc.set("A", speed)
        grayscale_data = px.get_grayscale_data()
        sc.set("D", grayscale_data )
        distance = px.get_distance()
        sc.set("F", distance)
        new_speak = sc.get('J')
        if new_speak in ["forward", "backward", "left", "right", "stop"]:
           speak = new_speak
        elif speak == "ping":
           pass
        else:  
           continue
        
      # used to save data during collection, but only collecting left and right
        if (speak != None and speak != prev_action and speak != "stop" and speak != "backward" and speak != "forward"):
            if save_frames(frame_id,speak):
                frame_id+= 1
              
        speed = sc.get("A")
        if speed is None:
           speed = 40
         

         #handles moving forward
        if speak in ["forward"]:
            print("prev action is", prev_action)
            if prev_action == "left":
               px.set_dir_servo_angle(5)
               print("im at point A")
            elif prev_action == "right":
               px.set_dir_servo_angle(-5)
               print("Im at point B")
            else:
               print("Im at point C")
                  
            px.forward(speed)
            prev_action = "forward"

         #handles moving backward
        elif speak in ["backward"]:
            if prev_action == "left":
               px.set_dir_servo_angle(5)
            elif prev_action == "right":
               
               px.set_dir_servo_angle(-5)
            else:
               px.set_dir_servo_angle(0)

            px.backward(speed)
            prev_action = "backward"

            
         #handles moving left
        elif speak in ["left"]:
            prev_action = "left"
            px.set_dir_servo_angle(-30)
            px.forward(30)
            if save_frames(frame_id,speak):
                  frame_id+= 1
               
         #moves right
        elif speak in ["right", "white", "rice"]:
            prev_action = "right"
            px.set_dir_servo_angle(20)
            px.forward(30)
            if save_frames(frame_id,speak):
                frame_id+= 1   
            
         #stops
        elif speak in ["stop"]:
            px.stop()
            
if __name__ == "__main__":
    try:
        main()
    finally:
        print("stop and exit")
        px.stop()
        Vilib.camera_close()





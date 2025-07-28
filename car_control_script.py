import websocket
import time
import json
import keyboard
import time
import threading

#creating a connection with the car in order to communicate
ws = websocket.create_connection("ws://192.168.1.8:8765")
last_command = ""

send_lock = threading.Lock()

#functions in order to ping the car to keep it active
def sendHeart():
    while True:
        with send_lock:
            ws.send(json.dumps({"Heart": "ping"}))
        time.sleep(5)
def pong():
    while True:
        response = ws.recv()

heart_thread = threading.Thread(target = sendHeart ,daemon = True)
heart_thread.start() 

pong_thread = threading.Thread(target = pong, daemon = True)
pong_thread.start() 


#function that sends the data to the robotic car so it can move
def sendMyCommand(command, speed = 40): 
    if speed is None:
        speed = 40 
    
    commandMessage = json.dumps(
        {"J": command,
         "A": speed  }) 

    with send_lock:
        ws.send(commandMessage)
    print(command)


#listens to keyboard input and sends it to car
while True:
    if (keyboard.is_pressed("w")):
        current = "forward"
    elif(keyboard.is_pressed("s")):
        current = "backward"
    elif(keyboard.is_pressed("a")):
        current = "left"
    elif(keyboard.is_pressed("d")):
        current = "right"
    else:
        current = "stop"
    
    if current != last_command:
        sendMyCommand(current)
        last_command = current
    
    time.sleep(0.1)
    





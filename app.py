import cv2
import numpy as np
import threading
import tkinter as tk
from tkinter import ttk
import requests
import pyttsx3
from geopy.geocoders import Nominatim

tts_engine = pyttsx3.init()
geolocator = Nominatim(user_agent="emergency_alert_system")

cap = cv2.VideoCapture(0)  # Capture from camera (0 for default)
EMERGENCY_NUMBER = 'YOUR_EMERGENCY_NUMBER'
GUI_ENABLED = True  # Set to False to disable GUI
GUI_REFRESH_INTERVAL_MS = 1000

if GUI_ENABLED:
    main_window = tk.Tk()
    main_window.title("Emergency Alert System")
    main_window.geometry("400x500")
    location_status = tk.StringVar(value="Location: Not Detected")
    sms_status = tk.StringVar(value="SMS Sending: Enabled")
    ip_status = tk.StringVar(value="IP Address: Not Detected")
    person_ratio = tk.StringVar(value="Person Ratio: Not Detected")
    heart_attack_status = tk.StringVar(value="Heart Attack Status: Not Detected")

net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
with open("coco.names", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

def fetch_ip_location():
    try:
        response = requests.get('http://ipinfo.io/json')
        response.raise_for_status()
        data = response.json()
        latitude, longitude = map(float, data.get('loc', "0,0").split(','))
        ip = data.get('ip', "Unknown IP")
        location = geolocator.reverse((latitude, longitude), language="en")
        return latitude, longitude, location.address if location else "Unknown Location", ip
    except requests.RequestException as e:
        print(f"Error fetching IP location: {e}")
        return None, None, None, None

def send_sms(message):
    # Simulate sending SMS (print to log)
    print(f"SMS Sent to {EMERGENCY_NUMBER}: {message}")
    sms_status.set(f"SMS Sent: {message}")

def process_camera_feed():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        detections = net.forward(output_layers)

        person_detected = False

        for detection in detections:
            if person_detected:
                break
            for object_info in detection:
                scores = object_info[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5 and class_names[class_id] == "person":
                    center_x, center_y, width, height = (object_info[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])).astype('int')
                    x = int(center_x - width / 2)
                    y = int(center_y - height / 2)

                    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                    cv2.putText(frame, f"{class_names[class_id]} {int(confidence * 100)}%", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    ratio = width / height if height > 0 else 0
                    person_ratio.set(f"Person Ratio: {ratio:.2f}")

                    if ratio > 1.5:
                        heart_attack_status.set("Heart Attack Status: Possible Heart Attack (Person Lying Down)")
                        latitude, longitude, address, ip = fetch_ip_location()
                        if latitude and longitude:
                            message = (f"Patient is currently experiencing a heart attack. Immediate medical assistance is required near {address}. "
                                       f"Coordinates: Lat: {latitude}, Long: {longitude}. IP: {ip}.")
                            tts_engine.say("You are currently experiencing a heart attack. Notifying emergency services.")
                            tts_engine.runAndWait()
                            send_sms(message)
                    else:
                        heart_attack_status.set("Heart Attack Status: Normal")

                    person_detected = True
                    break

        cv2.imshow("Camera Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def update_location_in_gui():
    latitude, longitude, address, ip = fetch_ip_location()
    if latitude and longitude:
        location_status.set(f"Location: {address} (Lat: {latitude}, Long: {longitude})")
        ip_status.set(f"IP Address: {ip}")
    else:
        location_status.set("Location: Not Detected")
        ip_status.set("IP Address: Not Detected")
    main_window.after(GUI_REFRESH_INTERVAL_MS, update_location_in_gui)

if GUI_ENABLED:
    ttk.Label(main_window, textvariable=location_status, wraplength=350).pack(pady=10)
    ttk.Label(main_window, textvariable=ip_status, wraplength=350).pack(pady=10)
    ttk.Label(main_window, textvariable=sms_status, wraplength=350).pack(pady=10)
    ttk.Label(main_window, textvariable=person_ratio, wraplength=350).pack(pady=10)
    ttk.Label(main_window, textvariable=heart_attack_status, wraplength=350).pack(pady=10)

camera_thread = threading.Thread(target=process_camera_feed, daemon=True)
camera_thread.start()

if GUI_ENABLED:
    update_location_in_gui()
    main_window.mainloop()
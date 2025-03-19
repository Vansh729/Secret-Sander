import cv2
import numpy as np
import os
import tkinter as tk
from twilio.rest import Client
import qrcode
from cryptography.fernet import Fernet

# ANSI escape codes
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RESET = "\033[0m"
MAGENTA = "\033[35m"
BOLD = "\033[1m"
GRAY_BACK = "\033[100m"
BLUE = "\033[94m"
BLINKING = "\033[5m"
green_BACK="\033[102m"

# path to save registered face data.
DATA_DIR = r'D:\project_faces'
os.makedirs(DATA_DIR, exist_ok=True)

# load pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# twilio account SID and token (use environment variables for security)
account_sid = "AC0e1002ad41e0430809cce9f78dd66988"
token = "a15b206e12abd2081a79ca9fc93f6ac8"
client = Client(account_sid, token)

# generate a key for AES encryption
key = Fernet.generate_key()
fernet = Fernet(key)

def detect_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return None
    (x, y, w, h) = faces[0]
    return gray[y:y+h, x:x+w]

def register_face(image, face_id):
    face = detect_face(image)
    if face is None:
        print(RED+"No face detected. Please try again."+RESET)
        return False

    # check if face ID already exists.
    if os.path.exists(os.path.join(DATA_DIR, f'{face_id}.jpg')):
        print(GREEN+f"Face ID '{face_id}' already exists. Please use a different ID."+RESET)
        return False

    # save the face image as a .jpg file
    face_image_path = os.path.join(DATA_DIR, f'{face_id}.jpg')
    cv2.imwrite(face_image_path, face)
    print(f"Face registered with ID: {face_id}")
    return True

def verify_face(image):
    face = detect_face(image)
    if face is None:
        print(RED+"No face detected. Please try again."+RESET)
        return False

    # load registered faces and their IDs
    registered_faces = []
    face_ids = []

    for filename in os.listdir(DATA_DIR):
        if filename.endswith('.jpg'):
            face_id = filename.split('.')[0]
            registered_face = cv2.imread(os.path.join(DATA_DIR, filename), cv2.IMREAD_GRAYSCALE)
            registered_faces.append(registered_face)
            face_ids.append(face_id)

    if not registered_faces:
        print(RED+"No registered faces found."+RESET)
        return False

    # create a face recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    # convert face_ids to a NumPy array
    labels = np.array(range(len(registered_faces)))

    # train the recognizer
    recognizer.train(registered_faces, labels)

    # predict the face
    label, confidence = recognizer.predict(face)

    # set a confidence threshold for validation
    confidence_threshold = 50  # Adjust this value as needed

    if confidence < confidence_threshold:
        print(YELLOW+f"Login successful! Recognized face ID: {face_ids[label]} with confidence: {confidence}"+RESET)
        secret_text()
        return True
    else:
        print(RED+f"Face mismatch! Confidence level: {confidence}. Try again."+RESET)
        return False

def generate_qr_code(data):
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(data)
    qr.make(fit=True)

    # create a PIL image from the QR code
    img = qr.make_image(fill_color="black", back_color="white")

    # convert the PIL image to a NumPy array for OpenCV
    img_np = np.array(img.convert("RGB"))  # convert to RGB format
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)  # convert RGB to BGR for OpenCV
    return img_np

def display_qr_code(image, duration_ms):
    cv2.imshow("QR Code", image)
    cv2.waitKey(duration_ms)  # wait for the specified duration
    cv2.destroyAllWindows()

def secret_text():
    # create the main window
    root = tk.Tk()
    root.title("WhatsApp Message Sender and QR Code Generator")

    # set the window size
    root.geometry("500x500")

    # QR Code Section
    qr_label = tk.Label(root, text="First register your mobile number with Twilio")
    qr_label.pack(pady=5)

    # function to handle the button click event for generating QR code
    def on_generate_qr_button_click():
        phone_number =+14155238886 
        data = f"http://wa.me/+14155238886?text=join%20feathers-lower"
        qr_code_image = generate_qr_code(data)
        display_qr_code(qr_code_image, 10000)

    qr_button = tk.Button(root, text="Generate QR Code", command=on_generate_qr_button_click)
    qr_button.pack(pady=10)

    # message Sender Section
    name_label = tk.Label(root, text="Name:")
    name_label.pack(pady=5)

    name_entry = tk.Entry(root, width=40)
    name_entry.pack(pady=5)

    num_label = tk.Label(root, text="Phone Number (with country code):")
    num_label.pack(pady=5)

    num_entry = tk.Entry(root, width=40)
    num_entry.pack(pady=5)

    mes_label = tk.Label(root, text="Message:")
    mes_label.pack(pady=5)

    mes_entry = tk.Entry(root, width=40)
    mes_entry.pack(pady=5)

    # function to send message via WhatsApp
    def send(num, mes):
        try:
            # encrypt the message
            encrypted_message = fernet.encrypt(mes.encode())
            print(BLUE+f"Encrypted Message: {encrypted_message.decode()}"+RESET)

            # send the encrypted message
            message = client.messages.create(
                from_="whatsapp:+14155238886", 
                body=encrypted_message.decode(), 
                to=f'whatsapp:{num}'
            )
            status_label.config(text="Message sent!", fg="green")
        except Exception as e:
            status_label.config(text="Error: " + str(e), fg="red")

    # function to handle the button click event for sending messages
    def on_send_button_click():
        name = name_entry.get()
        num = num_entry.get()
        mes = mes_entry.get()
        if name and num and mes:
            send(num, mes)
        else:
            status_label.config(text="Please fill in all fields.", fg="red")

    # send message button
    send_button = tk.Button(root, text="Send Encrypted Message", command=on_send_button_click)
    send_button.pack(pady=10)

    # status label to display messages
    status_label = tk.Label(root, text="", font=("Helvetica", 10))
    status_label.pack(pady=5)

    # decryption Section
    decrypt_label = tk.Label(root, text="Enter Encrypted Message to Decrypt:")
    decrypt_label.pack(pady=5)

    decrypt_entry = tk.Entry(root, width=40)
    decrypt_entry.pack(pady=5)

    # function to handle the button click event for decrypting messages
    def on_decrypt_button_click():
        encrypted_message = decrypt_entry.get()
        if encrypted_message:
            try:
                decrypted_message = fernet.decrypt(encrypted_message.encode()).decode()
                decrypt_status_label.config(text=f"Decrypted Message: {decrypted_message}", fg="green")
            except Exception as e:
                decrypt_status_label.config(text="Error: Invalid encrypted message.", fg="red")
        else:
            decrypt_status_label.config(text="Please enter an encrypted message.", fg="red")

    decrypt_button = tk.Button(root, text="Decrypt Message", command=on_decrypt_button_click)
    decrypt_button.pack(pady=10)

    # decryption status label
    decrypt_status_label = tk.Label(root, text="", font=("Helvetica", 10))
    decrypt_status_label.pack(pady=5)

    # run the GUI loop
    root.mainloop()

def CV_login_signin():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(RED+"Error: Could not open video."+RESET)
        return

    while True:
        print(GRAY_BACK + MAGENTA + BOLD + "<------ Choose an option ------>" + RESET)
        print(BLUE + "'1' to register your face" + RESET)
        print(BLUE + "'2' to login with face recognition" + RESET)
        print(BLUE + "'3' to exit" + RESET)
        user_input = input(BLINKING + "Enter your choice : " + RESET)
        
        if user_input == '3':
            print(RED+"Exiting program."+RESET)
            break

        success, image = cap.read()
        if not success:
            print(RED+"Ignoring empty camera frame."+RESET)
            continue

        if user_input == '1':
            face_id = input("Enter face ID to register: ").strip()
            register_face(image, face_id)
        elif user_input == '2':
            if verify_face(image):
                print(GREEN+"Face recognized successfully."+RESET)
            else:
                print(RED+"Face not recognized."+RESET)

        # display the image.
        cv2.imshow('Face Register/Login', cv2.flip(image, 1))

        if cv2.waitKey(5) & 0xFF == 27:  # press 'Esc' to exit.
            break

    cap.release()
    cv2.destroyAllWindows()
CV_login_signin()
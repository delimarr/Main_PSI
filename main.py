from tkinter.tix import IMAGETEXT
import customtkinter as customtkinter
from customtkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import cv2
import pandas as pd
from datetime import datetime
import os
import vision
import coordinate_system
import camera
from fanuc_py_xyw_chunk_debug import FanucError, Robot
from globals import callibration_lengt


# Import the global variable from the globals module
from globals import chip_quality_array
# ----------------------
#  Main Programm - GUI, Datenhandling
# Ersteller: S. Laube - SL
# Erstelldatum: 04.05.2024
# Änderungsdatum: 26.06.2024
# Version: 0.10 SL - GUI erstellt und Datenhandling von .csv und .xlsx - Funktion i. O
# Version: 0.20 SL - Kommunikation zu Fanuc, Wafer-Map und Vision Daten senden - Funktion i. O
# Version: 0.30 SL - GUI erweitert, Fenster für Error Nachrichten, Fenster für Wafer Bild - Funktion i. O.
# ----------------------

def update_scrollable_frame(message: str, color: str = "black"):
    # Get the current time and format it
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Add the timestamp to the message
    message_with_timestamp = f"{current_time} - {message}"
    label = customtkinter.CTkLabel(master=scrollable_frame, text=message_with_timestamp, text_color=color)
    label.pack(pady=5, padx=10)

def file_browser():
    print("Opening next UI")
    file = filedialog.askopenfile()
    if file:
        file_extension = os.path.splitext(file.name)[1]
        if file_extension == '.xlsx':
            df = pd.read_excel(file.name, header=None)
        elif file_extension == '.csv':
            df = pd.read_csv(file.name, header=None, delimiter=';')
        else:
            error_message = "Dateityp wird nicht unterstützt, .csv oder .xlsx verwenden."
            update_scrollable_frame(error_message, color="red")
            return

        waver_typ = df.iloc[0, 1]
        wafer = df.iloc[1, 1]
        current_datetime = datetime.now()
        data_array = df.iloc[7:].to_numpy()

        quality_values = [str(row[1]) for row in data_array]
        chip_quantity = len(quality_values)
        if len(quality_values)==36:
            update_scrollable_frame('Wafer map geladen', color='green')
        chip_quantity_ = f"{chip_quantity:02}"
        good_chip_count = len([row for row in data_array if row[1] != '0' and row[1] != 0])
        good_chip_count_ = f"{good_chip_count:02}"

        cmd = f"setregister:{chip_quantity_}:{good_chip_count_}:{':'.join(quality_values)}"
        
        # Information zu Qualität wird in VAR gespeichert
        quality_values = [str(row[1]) for row in data_array]
        global chip_quality_array
        chip_quality_array = quality_values 

        # Filter data_array for rows where the second column (Qualitaet) is 1
        if file_extension == '.xlsx':
            data_array = np.array([row for row in data_array if row[1] != 0])
        else:
            data_array = np.array([row for row in data_array if row[1] == '1'])


        ablagepunkt = [[0 for x in range(4)] for y in range(99)]
        for i in range(min(99, len(data_array))):
            for j in range(4):
                if i * 4 + j < len(data_array):
                    ablagepunkt[i][j] = data_array[i * 4 + j][0]

        ablagepunkt_indices = [(i, j) for i in range(99) for j in range(4) if ablagepunkt[i][j] != 0]
        ablagepunkt_index_1 = [index[0] + 1 for index in ablagepunkt_indices]
        ablagepunkt_index_2 = [index[1] + 1 for index in ablagepunkt_indices]

        waver_info_df = pd.DataFrame({
            'Waver Typ': [waver_typ] * len(data_array),
            'Wafer Nummer': [wafer] * len(data_array),
            'Chip Nummer': [row[0] for row in data_array],
            'Qualitaet': [row[1] for row in data_array],
            'Gel Pak Nummer': ablagepunkt_index_1,
            'Position auf Gel Pak': ablagepunkt_index_2
        })

        robot = Robot(
            robot_model="Fanuc",
            host="192.168.0.100",
            port=18735,
            ee_DO_type="RDO",
            ee_DO_num=7,
        )
        try:
            print('works')
            #response = robot.connect()
            update_scrollable_frame(response["msg"], "green" if response["success"] else "red")
        except:
            update_scrollable_frame('Roboter Verbindung fehlgeschlagen!', color="red")


        if response["success"]:
            try:
                response = robot.setregister(cmd=cmd)
                message = f"Antwort Code: {response['code']} und Nachricht: {response['msg']}"
                color = "green" if response['code'] == 0 else "red"
                update_scrollable_frame(message, color)
            except FanucError as e:
                update_scrollable_frame(str(e), color="red")
            finally:
                robot.disconnect()

        output_file_xlsx = f"{waver_typ}_Wafer_{wafer}_Ablage_{current_datetime.strftime('%Y-%m-%d_%H-%M-%S')}.xlsx"
        waver_info_df.to_excel(output_file_xlsx, index=False)

        output_file_csv = f"{waver_typ}_Wafer_{wafer}_Ablage_{current_datetime.strftime('%Y-%m-%d_%H-%M-%S')}.csv"
        waver_info_df.to_csv(output_file_csv, index=False)
        print(f"Files saved as {output_file_xlsx} and {output_file_csv}")

        return chip_quality_array


def vision_data():
    global chip_quality_array
    global indices
    
    vision_data=[]
    num_try = 0
    indices = [index for index, value in enumerate(chip_quality_array) if value == '1']

    
        
    #[
        #(1.0, 2.0, -3.0), (1.1, 20.1, 3.1), (1.2, 29.2, 0.2), (10.3, 2.3, 3.3),
        #(10.4, 20.4, -0.4), (10.5, 29.5, 3.5), (21.6, 2.6, 0.6), (21.7, 20.7, -0.7),
        #(21.8, 29.8, 0.8), (31.9, 2.9, 0.9), (32.0, 13.0, 1.0), (32.1, 23.1, 1.1),
        #(32.2, 33.2, 1.2), (42.3, 3.3, 1.3), (42.4, 23.4, 1.4), (32.5, 43.5, 1.5),
        #(2.6, 3.6, 1.6), (2.7, 3.7, 1.7), (2.8, 3.8, 1.8), (2.9, 3.9, 1.9),
        #(3.0, 4.0, 3.0), (3.1, 4.1, 2.1), (3.2, 4.2, 2.2), (3.3, 4.3, 2.3),
        #(3.4, 4.4, 2.4), (3.5, 4.5, 2.5), (3.6, 4.6, 2.6), (3.7, 4.7, 2.7),
        #(3.8, 4.8, 3.8), (3.9, 4.9, 3.9), (4.0, 5.0, 3.0), (4.1, 5.1, 3.1),
        #(4.2, 5.2, 3.2), (4.3, 5.3, 3.3), (4.4, 5.4, 3.4), (4.5, 5.5, 3.5)
    #]  # 36 tuples of (x, y, w)
    

    
    # Transform the vision_data points
    if chip_quality_array == []:
        update_scrollable_frame('Keine Wafer map geladen: Importiere zuerst eine Wafer map!', color="red")
    else:
        vision_data, detectSquareImg = vision.get_vision_data(indices)

        
    if len(vision_data) != 36:
        update_scrollable_frame(f'Anzahl Chips nicht correct: Aktuelle Zahl {len(vision_data)}', color="red")
    else:
        update_scrollable_frame('Es wurden alle 36 Chips gefunden!', color="green")
        print(f'visiondata:{vision_data}')        

    #print(f'vision data is: {vision_data}')
    #print(f'number of center: {len(vision_data)}')
    len_vision_data = len(vision_data)
    

    #if len_vision_data == 36:
    #    label = customtkinter.CTkLabel(master=frame, text=f"Wafer data: {str(vision_data)}")
    #    label.pack(pady=12, padx=10)
    # Convert the image data to a format compatible with ImageTk
    # Convert to OpenCV image
    try:
        # Resize the image
        PIL_image = Image.fromarray(detectSquareImg, 'RGB')
        ctk_image = ImageTk.PhotoImage(PIL_image)   
        image_label = customtkinter.CTkLabel(bottom_frame, image=ctk_image, text="")
        image_label.pack(padx=20, pady=20)
    except:
        print('no image data')

    robot = Robot(
        robot_model="Fanuc",
        host="192.168.0.100",
        port=18735,
        ee_DO_type="RDO",
        ee_DO_num=7,
    )
    print('Here')     

    try:
        #robot.connect()
        robot.send_vision_data(vision_data)
        robot.disconnect()
    except:
        update_scrollable_frame('Roboter Verbindung fehlgeschlagen', color="red")
        
   
    return vision_data, detectSquareImg


def coordinate_system_func():

    matrix, transformed_p4=coordinate_system.get_coordinateSystem()
    if transformed_p4[0]> callibration_lengt+0.3 or transformed_p4[0] <callibration_lengt - 0.3 and transformed_p4[1]> callibration_lengt+0.3 or transformed_p4[1] <callibration_lengt-0.3:
        update_scrollable_frame('Kallibrierung nicht exact: Bitte erneut versuchen!', color="red")
    else: 
        update_scrollable_frame('Kallibrierung Koordinatensystem i.O.', color="green")
    return matrix


def camera_setup():
    camera.camera_video()

customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("dark-blue")

root = customtkinter.CTk()
root.geometry("1100x900")

root.grid_rowconfigure(0, weight=0)  # Fixed height for upper frames
root.grid_rowconfigure(1, weight=1)  # Remaining space for bottom frame
root.grid_columnconfigure(0, weight=1)

main_container = customtkinter.CTkFrame(master=root, height=80)
main_container.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")

main_container.grid_columnconfigure(0, weight=1)
main_container.grid_columnconfigure(1, weight=6)

main_frame = customtkinter.CTkFrame(master=main_container)
main_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")

label = customtkinter.CTkLabel(master=main_frame, text="Import Wafer-Map", font=("Arial", 20))
label.pack(pady=12, padx=10)

button = customtkinter.CTkButton(master=main_frame, text="Öffnen", command=file_browser)
button.pack(pady=12, padx=10)

label = customtkinter.CTkLabel(master=main_frame, text="Import Vision Daten", font=("Arial", 20))
label.pack(pady=12, padx=10)

vision_button = customtkinter.CTkButton(master=main_frame, text="Vision Data", command=vision_data)
vision_button.pack(pady=12, padx=10)

label = customtkinter.CTkLabel(master=main_frame, text="Settings", font=("Arial", 20))
label.pack(pady=12, padx=10)

coordinat_button = customtkinter.CTkButton(master=main_frame, text="Kalibration", command=coordinate_system_func)
coordinat_button.pack(pady=12, padx=10)

coordinat_button = customtkinter.CTkButton(master=main_frame, text="Kamera", command=camera_setup)
coordinat_button.pack(pady=12, padx=10)

scrollable_frame = customtkinter.CTkScrollableFrame(master=main_container)
scrollable_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")

bottom_frame = customtkinter.CTkFrame(master=root)
bottom_frame.grid(row=1, column=0, padx=20, pady=20, sticky="nsew")

bottom_label = customtkinter.CTkLabel(master=bottom_frame, text="Wafer Bild", font=("Arial", 20))
bottom_label.pack(pady=12, padx=5)


root.mainloop()
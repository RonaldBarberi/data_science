# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 10:59:00 2025

@author: Ronald.Barberi (KretoN)
"""

#%% Imported libraries

import os
import csv
import cv2
import glob
import time
import psutil
import joblib
import numpy as np
import pandas as pd
import mediapipe as mp
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#%% Create class

class DPCommandFigure:
    def __init__(self, dic_args):
        self.dic_args = dic_args

    def generate_dataset_to_test(self):
        while True:
            new_dataset = input('¿Deseas agregar algun nuevo dataset? (s/n):').lower()
            if new_dataset == 's':
                os.makedirs(self.dic_args['path_save'], exist_ok=True)
                mp_hands = mp.solutions.hands
                hands = mp_hands.Hands(static_image_mode=False, max_num_hands=self.dic_args['hands'])
                mp_drawing = mp.solutions.drawing_utilss
                cap = cv2.VideoCapture(0)

                while True:
                    file_name = input('Escribe el nombre del primer dataset:')
                    full_path = f"{self.dic_args['path_save']}/{file_name}.csv"
                    amount_save = 0

                    with open(full_path, mode='w', newline='') as f:
                        writer = csv.writer(f)
                        print('Press S to save reference.')
                        print('Press R to Exit.')

                        while True:
                            ret, frame = cap.read()
                            if not ret:
                                break
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            results = hands.process(frame_rgb)
                            if results.multi_hand_landmarks:
                                for hand_landmarks in results.multi_hand_landmarks:
                                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                                    landmarks = []
                                    for lm in hand_landmarks.landmark:
                                        landmarks.extend([lm.x, lm.y, lm.z])

                                    if cv2.waitKey(1) & 0xFF == ord('s'):
                                        writer.writerow(landmarks)
                                        print(f'Save {amount_save}')
                                        amount_save += 1

                            cv2.imshow("Capture of signs", frame)
                            if cv2.waitKey(1) & 0xFF == ord('r'):
                                break

                    cont = input('¿Deseas capturar otro archivo? (s/n): ').lower()
                    if cont != 's':
                        print('Proceso finalizado.')
                        break

                cap.release()
                cv2.destroyAllWindows()
                cv2.waitKey(1)
                break
            elif new_dataset == 'n':
                pass
                break
            else:
                print('Opcion no valida')
    

    def group_all_dataset(self):
        files = glob.glob(os.path.join(self.dic_args['path_save'], '*.csv'))
        datos = []

        for file in files:
            nombre_clase = os.path.splitext(os.path.basename(file))[0]
            df = pd.read_csv(file, header=None)
            df['label'] = nombre_clase
            datos.append(df)

        self.df_total = pd.concat(datos, ignore_index=True)
        self.df_total.to_csv(os.path.join(self.dic_args['path_save'], 'full_dataset.csv'), index=False)
        print(self.df_total.head())


    def training_model(self):
        X = self.df_total.drop('label', axis=1)
        y = self.df_total['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        clf = RandomForestClassifier(n_estimators=1_000, random_state=42)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        print(classification_report(y_test, y_pred))

        cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
        disp.plot(cmap='Blues', xticks_rotation='vertical')
        plt.title('Matriz de Confusión del Modelo Entrenado')
        plt.tight_layout()
        plt.show()
        
        joblib.dump(clf, 'model_train.pkl')


    def go_model_scan(self):
        modelo = joblib.load('model_train.pkl')

        acciones = {
            'pulgar': lambda: self.one_fuction_random(self.dic_args['path_one'], 6),
            'chill': lambda: self.one_fuction_random(self.dic_args['path_two'], 7),
            'dedomedio': lambda: os.system('shutdown /r /t 0'),
            # 'dedomedio': lambda: print('shutdown /r /t 0'),
        }

        # Inicializar MediaPipe Hands
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(static_image_mode=False, max_num_hands=self.dic_args['hands'])
        mp_drawing = mp.solutions.drawing_utils

        # Abrir webcam
        cap = cv2.VideoCapture(0)

        print("[INFO] Presiona 'q' para salir")
        last_prediction_time = 0
        prediction_interval = 0.5

        accion_en_espera = None
        tiempo_inicio_espera = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])

                    if len(landmarks) == 63:
                        current_time = time.time()
                        if current_time - last_prediction_time >= prediction_interval:
                            last_prediction_time = current_time

                            X_input = np.array(landmarks).reshape(1, -1)
                            pred = modelo.predict(X_input)[0]

                            if pred in acciones:
                                if accion_en_espera is None:
                                    accion_en_espera = pred
                                    tiempo_inicio_espera = current_time
                                elif pred == accion_en_espera:
                                    if current_time - tiempo_inicio_espera >= 1:
                                        acciones[pred]()
                                        accion_en_espera = None
                                else:
                                    accion_en_espera = pred
                                    tiempo_inicio_espera = current_time

            else:
                accion_en_espera = None

            cv2.imshow('Reconocimiento de señas', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


    def one_fuction_random(self, path_movie, time_movie):
        duracion_video = 1

        procesos_antes = {p.pid: p.name() for p in psutil.process_iter()}
        os.startfile(path_movie)
        time.sleep(duracion_video)

        procesos_despues = {p.pid: p.name() for p in psutil.process_iter()}
        procesos_nuevos = set(procesos_despues.items()) - set(procesos_antes.items())
        time.sleep(time_movie - duracion_video)
        for pid, name in procesos_nuevos:
            try:
                os.system(f'taskkill /pid {pid} /f')
            except Exception as e:
                print(e)


    def main(self):
        self.generate_dataset_to_test()
        self.group_all_dataset()
        self.training_model()
        self.go_model_scan()


#%% Use class

if __name__ == "__main__":
    current_path = os.path.abspath(os.path.dirname(__file__))
    dic_args = {
        'hands': 2,
        'path_save': os.path.join(current_path, '..', 'data'),
        'path_one': os.path.join(current_path, '..', 'data', 'chill_cojones.mp4'),
        'path_two': os.path.join(current_path, '..', 'data', 'perro_bailando.mp4'),
    }
    deepL = DPCommandFigure(dic_args)
    deepL.main()

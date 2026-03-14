import tkinter as tk
from tkinter import filedialog, messagebox, font
import cv2
from PIL import Image, ImageTk
import os
import mediapipe as mp

from predict_video import predict_video



def get_ground_truth(video_path):

    parts = os.path.normpath(video_path).split(os.sep)

    labels = {
        "tripping",
        "high_sticking",
        "holding",
        "cross_checking",
        "interference"
    }

    for p in parts:
        if p in labels:
            return p

    return "unknown"


class GestureApp:

    def __init__(self, root):

        self.root = root
        self.root.title("Rozpoznawanie gestów sędziowskich – hokej")
        self.root.geometry("800x750")
        self.root.configure(bg="#e6f4ff")

        self.video_path = None
        self.cap = None

        # MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_draw = mp.solutions.drawing_utils

        self.build_ui()


    
    # ui
    def build_ui(self):

        title_font = font.Font(family="Arial", size=16, weight="bold")
        text_font = font.Font(family="Arial", size=11)
        result_font = font.Font(family="Arial", size=13, weight="bold")

        # title
        tk.Label(
            self.root,
            text="System rozpoznawania gestów sędziów hokejowych",
            bg="#e6f4ff",
            fg="#003366",
            font=title_font
        ).pack(pady=10)

        # buttons
        buttons_frame = tk.Frame(self.root, bg="#e6f4ff")
        buttons_frame.pack(pady=10)

        self.select_btn = tk.Button(
            buttons_frame,
            text="📁 Wybierz plik wideo",
            bg="#87ceeb",
            fg="#003366",
            font=text_font,
            width=30,
            command=self.select_file
        )
        self.select_btn.pack(pady=5)

        self.predict_btn = tk.Button(
            buttons_frame,
            text="▶️ Rozpoznaj gest",
            bg="#b0e0e6",
            fg="#003366",
            font=text_font,
            width=30,
            state="disabled",
            command=self.run_prediction
        )
        self.predict_btn.pack(pady=5)

        # file name
        self.filename_label = tk.Label(
            self.root,
            text="Nie wybrano pliku",
            bg="#e6f4ff",
            fg="#003366",
            font=text_font
        )
        self.filename_label.pack(pady=5)

        # video
        self.video_label = tk.Label(
            self.root,
            bg="#cfe9ff",
            width=640,
            height=360
        )
        self.video_label.pack(pady=15)

        # result
        self.result_label = tk.Label(
            self.root,
            text="Wynik: —",
            bg="#e6f4ff",
            fg="#003366",
            font=result_font,
            justify="center"
        )
        self.result_label.pack(pady=15)


    # file selection

    def select_file(self):

        file_path = filedialog.askopenfilename(
            title="Wybierz plik wideo",
            filetypes=[("Pliki wideo", "*.mp4 *.avi *.mov *.mkv")]
        )

        if file_path:

            self.video_path = file_path

            self.filename_label.config(
                text=f"Wybrany plik: {os.path.basename(file_path)}"
            )

            self.predict_btn.config(state="normal")

            self.start_video()

            self.result_label.config(
                text="Wynik: —",
                fg="#003366"
            )


    # video preview

    def start_video(self):

        if self.cap:
            self.cap.release()

        self.cap = cv2.VideoCapture(self.video_path)

        self.update_frame()


    def update_frame(self):

        if not self.cap:
            return

        ret, frame = self.cap.read()

        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.update_frame()
            return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.pose.process(rgb)

        if results.pose_landmarks:

            landmarks = results.pose_landmarks.landmark

            hand_points = [11,12,13,14,15,16]

            h, w, _ = frame.shape

            for idx in hand_points:

                lm = landmarks[idx]

                x = int(lm.x * w)
                y = int(lm.y * h)

                cv2.circle(frame, (x,y), 6, (0,255,0), -1)

            connections = [
                (11,13),(13,15),
                (12,14),(14,16),
                (11,12)
            ]

            for c in connections:

                p1 = landmarks[c[0]]
                p2 = landmarks[c[1]]

                x1,y1 = int(p1.x*w), int(p1.y*h)
                x2,y2 = int(p2.x*w), int(p2.y*h)

                cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),2)

        frame = cv2.resize(frame, (640, 360))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)

        self.video_label.imgtk = imgtk
        self.video_label.config(image=imgtk)

        self.root.after(30, self.update_frame)


    #prediction

    def run_prediction(self):

        if not self.video_path:
            messagebox.showwarning("Błąd", "Najpierw wybierz plik wideo.")
            return

        self.result_label.config(text="Analiza wideo... ", fg="#003366")
        self.root.update()

        try:

            predicted, confidence = predict_video(self.video_path)

            ground_truth = get_ground_truth(self.video_path)

            gesture_map = {
                "tripping": "TRIPPING – podstawienie",
                "high_sticking": "HIGH STICKING – gra wysoko uniesionym kijem",
                "holding": "HOLDING – przytrzymywanie",
                "cross_checking": "CROSS CHECKING",
                "interference": "INTERFERENCE"
            }

            pred_txt = gesture_map.get(predicted, predicted)
            gt_txt = gesture_map.get(ground_truth, ground_truth)

            correct = predicted == ground_truth

            if correct:
                color = "#2e8b57"
            else:
                color = "#b22222"

            self.result_label.config(
                text=(
                    f"Predykcja modelu:\n{pred_txt}\n"
                    f"Pewność modelu: {confidence*100:.1f}%\n\n"
                    f"Prawdziwa klasa:\n{gt_txt}\n"
                ),
                fg=color
            )

        except Exception as e:

            messagebox.showerror(
                "Błąd",
                "Wystąpił błąd podczas analizy wideo."
            )

            print(e)


if __name__ == "__main__":

    root = tk.Tk()
    app = GestureApp(root)
    root.mainloop()






import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk
import os

from main import jalankan_training, jalankan_pencocokan

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class FaceRecognitionApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Face Recognition - EigenFace")
        self.geometry("1300x750")
        self.resizable(False, False)

        self.dataset_path = ""
        self.image_path = ""
        self.model = None

        # TITLE
        self.title_label = ctk.CTkLabel(
            self, text="Face Recognition System", font=("Arial", 28, "bold")
        )
        self.title_label.pack(pady=20)

        # MAIN FRAME
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=10)

        # LEFT FRAME
        self.left_frame = ctk.CTkFrame(self.main_frame, width=300)
        self.left_frame.pack(side="left", fill="y", padx=15, pady=15)

        # RIGHT FRAME
        self.right_frame = ctk.CTkFrame(self.main_frame)
        self.right_frame.pack(side="right", fill="both", expand=True, padx=15, pady=15)

        # DATASET BUTTON
        self.dataset_button = ctk.CTkButton(
            self.left_frame, text="Choose Dataset Folder",
            command=self.choose_dataset, height=45, font=("Arial", 15, "bold")
        )
        self.dataset_button.pack(pady=(30, 15), padx=20, fill="x")

        self.dataset_label = ctk.CTkLabel(
            self.left_frame, text="No dataset selected", wraplength=250, justify="left"
        )
        self.dataset_label.pack(pady=(0, 20), padx=20)

        # TRAIN BUTTON
        self.train_button = ctk.CTkButton(
            self.left_frame, text="Train Model", command=self.run_training,
            height=45, font=("Arial", 15, "bold"),
            fg_color="#2563eb", hover_color="#1d4ed8"
        )
        self.train_button.pack(pady=8, padx=20, fill="x")

        self.train_status = ctk.CTkLabel(
            self.left_frame, text="Model: Belum di-training", wraplength=250, justify="left"
        )
        self.train_status.pack(pady=(0, 10), padx=20)

        # IMAGE BUTTON
        self.image_button = ctk.CTkButton(
            self.left_frame, text="Upload Test Image",
            command=self.choose_image, height=45, font=("Arial", 15, "bold")
        )
        self.image_button.pack(pady=8, padx=20, fill="x")

        self.image_label = ctk.CTkLabel(
            self.left_frame, text="No image selected", wraplength=250, justify="left"
        )
        self.image_label.pack(pady=(0, 20), padx=20)

        # RECOGNIZE BUTTON
        self.recognize_button = ctk.CTkButton(
            self.left_frame, text="Recognize Face", command=self.recognize_face,
            height=50, font=("Arial", 16, "bold"),
            fg_color="#16a34a", hover_color="#15803d"
        )
        self.recognize_button.pack(pady=20, padx=20, fill="x")

        # RESULT AREA
        self.result_title = ctk.CTkLabel(
            self.left_frame, text="Recognition Result", font=("Arial", 18, "bold")
        )
        self.result_title.pack(pady=(10, 5))

        self.result_text = ctk.CTkTextbox(
            self.left_frame, width=250, height=250, font=("Arial", 14)
        )
        self.result_text.pack(padx=20, pady=10)
        self.result_text.insert("0.0", "Result will appear here...")

        # =====================================
        # RIGHT FRAME - COMPARISON AREA
        # =====================================
        self.preview_title = ctk.CTkLabel(
            self.right_frame, text="Face Comparison", font=("Arial", 20, "bold")
        )
        self.preview_title.pack(pady=(15, 5))

        self.compare_frame = ctk.CTkFrame(self.right_frame, fg_color="transparent")
        self.compare_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Kolom kiri - foto test
        self.col_test = ctk.CTkFrame(self.compare_frame)
        self.col_test.pack(side="left", fill="both", expand=True, padx=10)

        ctk.CTkLabel(self.col_test, text="Test Image", font=("Arial", 15, "bold")).pack(pady=(10, 5))

        self.image_test_preview = ctk.CTkLabel(
            self.col_test, text="No Image", width=280, height=320,
            fg_color=("gray20", "gray20"), corner_radius=12
        )
        self.image_test_preview.pack(pady=5, padx=10)

        self.label_test_name = ctk.CTkLabel(
            self.col_test, text="—", font=("Arial", 13), text_color="gray"
        )
        self.label_test_name.pack(pady=5)

        # Kolom tengah - VS + persentase
        self.col_mid = ctk.CTkFrame(self.compare_frame, fg_color="transparent", width=90)
        self.col_mid.pack(side="left", fill="y", padx=5)
        self.col_mid.pack_propagate(False)

        self.vs_label = ctk.CTkLabel(
            self.col_mid, text="VS", font=("Arial", 22, "bold"), text_color="gray"
        )
        self.vs_label.place(relx=0.5, rely=0.33, anchor="center")

        self.similarity_label = ctk.CTkLabel(
            self.col_mid, text="", font=("Arial", 15, "bold"), text_color="#16a34a"
        )
        self.similarity_label.place(relx=0.5, rely=0.50, anchor="center")

        self.status_icon = ctk.CTkLabel(
            self.col_mid, text="", font=("Arial", 30, "bold")
        )
        self.status_icon.place(relx=0.5, rely=0.65, anchor="center")

        # Kolom kanan - foto match
        self.col_match = ctk.CTkFrame(self.compare_frame)
        self.col_match.pack(side="left", fill="both", expand=True, padx=10)

        ctk.CTkLabel(self.col_match, text="Closest Match", font=("Arial", 15, "bold")).pack(pady=(10, 5))

        self.image_match_preview = ctk.CTkLabel(
            self.col_match, text="No Match", width=280, height=320,
            fg_color=("gray20", "gray20"), corner_radius=12
        )
        self.image_match_preview.pack(pady=5, padx=10)

        self.label_match_name = ctk.CTkLabel(
            self.col_match, text="—", font=("Arial", 13), text_color="gray"
        )
        self.label_match_name.pack(pady=5)

    # =====================================
    # CHOOSE DATASET
    # =====================================
    def choose_dataset(self):
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.dataset_path = folder_selected
            self.dataset_label.configure(text=f"Dataset Selected:\n{folder_selected}")

    # =====================================
    # CHOOSE IMAGE
    # =====================================
    def choose_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.png *.jpg *.jpeg")]
        )
        if file_path:
            self.image_path = file_path
            filename = os.path.basename(file_path)
            self.image_label.configure(text=f"Image Selected:\n{filename}")
            self.display_test_image(file_path)

    # =====================================
    # RUN TRAINING
    # =====================================
    def run_training(self):
        if not self.dataset_path:
            self.show_result("Pilih dataset folder dulu!")
            return

        self.show_result("Training sedang berjalan...\nTunggu sebentar...")
        self.update()

        try:
            import main as backend
            backend.DATASET_FOLDER = self.dataset_path
            self.model = jalankan_training(paksa_ulang=False)

            if self.model:
                self.train_status.configure(text="Model: Siap ✓")
                self.show_result("Training selesai!\nModel siap digunakan.")
            else:
                self.show_result("Training gagal!\nCek dataset kamu.")

        except Exception as e:
            self.show_result(f"Error saat training:\n{str(e)}")

    # =====================================
    # DISPLAY TEST IMAGE
    # =====================================
    def display_test_image(self, image_path):
        image = Image.open(image_path)
        image.thumbnail((280, 320))
        photo = ctk.CTkImage(light_image=image, dark_image=image, size=image.size)
        self.image_test_preview.configure(image=photo, text="")
        self.image_test_preview.image = photo
        self.label_test_name.configure(text=os.path.basename(image_path), text_color="white")

    # =====================================
    # DISPLAY MATCH IMAGE
    # =====================================
    def display_match_image(self, image_path, label):
        try:
            image = Image.open(image_path)
            image.thumbnail((280, 320))
            photo = ctk.CTkImage(light_image=image, dark_image=image, size=image.size)
            self.image_match_preview.configure(image=photo, text="")
            self.image_match_preview.image = photo
            self.label_match_name.configure(text=label, text_color="white")
        except Exception:
            self.image_match_preview.configure(image=None, text="Gambar tidak ditemukan")

    # =====================================
    # HITUNG PERSENTASE KEMIRIPAN
    # =====================================
    def hitung_similarity(self, jarak, threshold):
        if jarak <= 0:
            return 100.0
        persen = max(0.0, (1 - (jarak / threshold)) * 100)
        return round(persen, 1)

    # =====================================
    # RECOGNIZE FUNCTION
    # =====================================
    def recognize_face(self):
        if not self.dataset_path:
            self.show_result("Pilih dataset folder dulu!")
            return
        if not self.image_path:
            self.show_result("Upload gambar test dulu!")
            return
        if self.model is None:
            self.show_result("Lakukan Training dulu\nsebelum mengenali wajah!")
            return

        try:
            import main as backend
            threshold = backend.THRESHOLD
            hasil = jalankan_pencocokan(self.model, self.image_path)
            similarity = self.hitung_similarity(hasil['jarak'], threshold)

            if hasil['dikenali']:
                result_message = (
                    f"WAJAH DIKENALI!\n\n"
                    f"Nama      : {hasil['label']}\n"
                    f"Kemiripan : {similarity}%\n"
                    f"Jarak     : {hasil['jarak']:.4f}\n"
                    f"Pesan     : {hasil['pesan']}"
                )
                self.similarity_label.configure(text=f"{similarity}%", text_color="#16a34a")
                self.status_icon.configure(text="✓", text_color="#16a34a")
                if hasil.get('path_gambar'):
                    self.display_match_image(hasil['path_gambar'], hasil['label'])
            else:
                result_message = (
                    f"TIDAK DIKENALI\n\n"
                    f"Kemiripan : {similarity}%\n"
                    f"Jarak     : {hasil['jarak']:.4f}\n"
                    f"Pesan     : {hasil['pesan']}"
                )
                self.similarity_label.configure(text=f"{similarity}%", text_color="#dc2626")
                self.status_icon.configure(text="✗", text_color="#dc2626")
                if hasil.get('path_gambar'):
                    self.display_match_image(hasil['path_gambar'], "Tidak cocok")
                else:
                    self.image_match_preview.configure(image=None, text="No Match")
                    self.label_match_name.configure(text="—", text_color="gray")

            self.show_result(result_message)

        except Exception as e:
            self.show_result(f"Error:\n{str(e)}")

    # =====================================
    # SHOW RESULT
    # =====================================
    def show_result(self, message):
        self.result_text.delete("0.0", "end")
        self.result_text.insert("0.0", message)


# ==========================================
# RUN APPLICATION
# ==========================================
if __name__ == "__main__":
    app = FaceRecognitionApp()
    app.mainloop()

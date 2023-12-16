

#MAIN APP

import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the pre-trained InceptionV3 model
model = InceptionV3(weights='imagenet')

class StockPredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Stock Prediction App")

        # Label for additional message
        self.additional_message_label = tk.Label(root, text="If it's not a clear financial chart or line graph, the score may be inaccurate and invalid.")
        self.additional_message_label.pack(pady=5)

        self.image_label = tk.Label(root)
        self.image_label.pack(pady=10)

        self.upload_button = tk.Button(root, text="Upload Image", command=self.upload_image)
        self.upload_button.pack(pady=10)

        self.predict_button = tk.Button(root, text="Predict", command=self.predict_stock)
        self.predict_button.pack(pady=10)

        self.prediction_label = tk.Label(root, text="")
        self.prediction_label.pack(pady=10)

        # Create a Text widget to display the disclaimer text in a box
        self.disclaimer_text_box = tk.Text(root, wrap=tk.WORD, height=10, width=40)
        self.disclaimer_text_box.insert(tk.END, self.get_disclaimer_text())
        self.disclaimer_text_box.pack(pady=10)

    def get_disclaimer_text(self):
        return (
            "Ensure the uploaded image represents a chart for accurate predictions. "
            "The rating score is a representation of the confidence level in the prediction. "
            "This application does not offer financial advice, and any predictions made should not be construed as such. "
            "Users are reminded that the application's output is solely based on the provided image "
            "and should exercise caution and seek professional financial advice for any investment decisions. "
            "By using this application, users acknowledge and agree to these terms."
        )

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.display_image(file_path)

    def display_image(self, file_path):
        img = Image.open(file_path)
        img = img.resize((300, 300))
        img = ImageTk.PhotoImage(img)
        self.image_label.config(image=img)
        self.image_label.image = img
        self.file_path = file_path

    def is_chart(self, predictions):
        # Map class names to their indices
        class_names_to_indices = {
            "business_school": 874,
            "financial_news": 795,
            "stock_index": 856,
            "face": 835  # Change the class index for face
        }

        # Use the indices to get the probabilities
        chart_probabilities = [predictions[0, class_names_to_indices[class_name]] for class_name in class_names_to_indices]

        # Set a higher threshold for financial chart class
        chart_threshold = np.max(chart_probabilities) - 0.2  # Adjust the threshold as needed

        return predictions[0, class_names_to_indices["stock_index"]] > chart_threshold

    def preprocess_image(self):
        img = image.load_img(self.file_path, target_size=(299, 299))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return img_array

    def predict_stock(self):
        if hasattr(self, 'file_path'):
            processed_image = self.preprocess_image()
            predictions = model.predict(processed_image)
            if self.is_chart(predictions):
                self.display_prediction(predictions)
            else:
                self.prediction_label.config(text="Upload a clearer image of a financial chart.")

    def calculate_rating_score(self, prediction_prob):
        # Map the prediction probability to a rating score between 1 and 100
        rating_score = int(prediction_prob * 100)
        return rating_score

    def display_prediction(self, predictions):
        # Use the maximum probability from the predictions array
        prediction_prob = np.max(predictions)

        # Adjust thresholds based on your model's output and desired criteria
        buy_threshold = 0.6
        hold_threshold = 0.4

        if prediction_prob >= buy_threshold:
            prediction_text = "BUY"
        elif prediction_prob >= hold_threshold:
            prediction_text = "HOLD"
        else:
            prediction_text = "SELL"

        # Calculate the rating score
        rating_score = self.calculate_rating_score(prediction_prob)

        # Add the rating score to the prediction result
        rating_score_text = f"Rating Score: {rating_score}/100"

        # Combine the prediction result, rating score, and disclaimer
        result_with_disclaimer = f"Prediction: {prediction_text}\n{rating_score_text}"

        self.prediction_label.config(text=result_with_disclaimer)

if __name__ == "__main__":
    root = tk.Tk()
    app = StockPredictionApp(root)
    root.mainloop()


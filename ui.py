import tkinter as tk
from tkinter import ttk
from predict import predict_bias


def on_predict_button_click():
    input_sentence = sentence_entry.get()
    probability, category = predict_bias(input_sentence)
    result_label.config(
        text=f"The input sentence has a {probability * 100:.2f}% probability of being in the '{category}' category.")


# Create the main window
root = tk.Tk()
root.title("Bias Prediction")

# Create a frame to hold the input widgets
frame = ttk.Frame(root, padding="10")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

# Create a label and entry for the input sentence
sentence_label = ttk.Label(frame, text="Enter the sentence:")
sentence_label.grid(row=0, column=0, sticky=tk.W)
sentence_entry = ttk.Entry(frame, width=50)
sentence_entry.grid(row=1, column=0, sticky=(tk.W, tk.E))

# Create a button to predict the probability of bias
predict_button = ttk.Button(
    frame, text="Predict Bias", command=on_predict_button_click)
predict_button.grid(row=2, column=0, pady=10)

# Create a label to display the result
result_label = ttk.Label(frame, text="")
result_label.grid(row=3, column=0, sticky=tk.W)

# Set column weight to make the entry widget expand horizontally
frame.columnconfigure(0, weight=1)

# Start the Tkinter event loop
root.mainloop()

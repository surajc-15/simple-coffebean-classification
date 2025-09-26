# Coffee Bean Classification with AI

This project provides an end-to-end pipeline for classifying coffee beans (Dark, Green, Light, Medium) using a Convolutional Neural Network (CNN) and a user-friendly Streamlit web app. It empowers small-scale farmers with instant, objective grading of coffee beans for fairer pricing and improved quality control.

**GitHub Repository:**  
https://github.com/surajc-15/simple-coffebean-classification

---

## Features

- **CNN-based image classifier** for coffee bean types.
- **Streamlit web app** for easy image upload and instant prediction.
- **Clear grading report** for negotiation and transparency.
- **Easy-to-follow training pipeline** for your own dataset.

---

## Folder Structure

```
CoffeBeanClassification/
│
├── train/                # Training images (subfolders: Dark, Green, Light, Medium)
├── test/                 # Test images (same subfolders)
├── app.py                # Streamlit web app
├── train_model.py        # Model training script
├── requirements.txt      # Python dependencies
├── .gitignore
└── README.md
```

---

## Step-by-Step Instructions

### 1. Clone the Repository

```sh
git clone https://github.com/surajc-15/simple-coffebean-classification.git
cd simple-coffebean-classification
```

### 2. Prepare Your Data

- Place your images in `train/` and `test/` folders, each with subfolders: `Dark`, `Green`, `Light`, `Medium`.
- Example: `train/Dark/dark1.png`, `test/Green/green1.png`, etc.

### 3. Create and Activate a Virtual Environment

```sh
python -m venv coffeenv
# On Windows:
coffeenv\Scripts\activate
# On Mac/Linux:
# source coffeenv/bin/activate
```

### 4. Install Dependencies

```sh
pip install -r requirements.txt
```

### 5. Train the Model

```sh
python train_model.py
```

- This will train the CNN and save `coffee_bean_classifier.h5` in your project folder.

### 6. Run the Streamlit Web App

```sh
streamlit run app.py
```

- Open the provided local URL in your browser.
- Upload a coffee bean image and view the predicted class and grade.

---

## Notes

- Adjust image size and model parameters in `train_model.py` as needed for your dataset.
- The `.gitignore` file excludes model weights, environment folders, and temporary files from version control.

---

## License

This project is for educational and non-commercial use. Please cite the repository if you use it in your work.

---

## Acknowledgements

- Built with [TensorFlow](https://www.tensorflow.org/), [Streamlit](https://streamlit.io/), [Pillow](https://python-pillow.org/), and [Plotly](https://plotly.com/python/).

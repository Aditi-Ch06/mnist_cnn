🖊️ Handwritten Digit Prediction (MNIST)

A deep learning project to classify handwritten digits (0–9) using a Convolutional Neural Network (CNN) implemented in PyTorch. The model is trained on the MNIST dataset and deployed with a Flask web app for real-time predictions.

🚀 Project Overview

Dataset: MNIST
 (60,000 training + 10,000 testing images).

Model: CNN built with PyTorch.

Goal: Predict the digit (0–9) from an image of handwritten digits.

Deployment: Flask-based web service.

🔗 Live Demo: [Click here to try the deployed service](https://mnist-cnn-1.onrender.com)

📂 Project Structure

├── data/MNIST/raw                # MNIST dataset 

├── templates/

│   └──index.html

├── .gitignore

├── Dockerfile

├── app.py               # Deployment code (Flask)

├── mnist_cnn.pth

├── render.yaml

├── runtime.txt

├── requirements.txt     # Dependencies

└── README.md            # Project documentation

🧠 Model Architecture

The CNN architecture defined in app.py:

Convolutional Layer 1: 1 input channel → 32 output channels, kernel size 5, stride 1, padding = same

Activation: ReLU

MaxPooling Layer 1: kernel size 2, stride 2

Convolutional Layer 2: 32 input channels → 64 output channels, kernel size 5, stride 1, padding = same

Activation: ReLU

MaxPooling Layer 2: kernel size 2, stride 2

Flatten Layer

Fully Connected Layer 1: 64×7×7 → 1024 units

Activation: ReLU

Fully Connected Layer 2: 1024 → 10 (digit classes)

📊 Results

Training Accuracy: ~99%

Test Accuracy: ~98%

Model shows good generalization with minimal overfitting.

Deployed Flask app can take an uploaded image and return the predicted digit.

⚙️ Installation & Usage
1. Clone the repo
git clone https://github.com/Aditi-Ch06/mnist_cnn.git
cd mnist-digit-prediction

2. Install dependencies
pip install -r requirements.txt

3. Run Flask app locally
python app.py


Then open your browser at http://127.0.0.1:8000/

🌐 Deployment

The app is deployed using Flask and accessible at:
👉 https://mnist-cnn-1.onrender.com

📸 Screenshots

<img width="1913" height="825" alt="image" src="https://github.com/user-attachments/assets/050f707b-4a04-4036-a7ba-b630dfd382a1" />


🔮 Future Improvements

Add a drawing canvas to directly write digits in the browser.

Extend model to handle non-MNIST handwriting samples.

Deploy on Hugging Face Spaces or Heroku for wider access.

🤝 Contributing

Pull requests are welcome! For significant changes, open an issue first to discuss what you’d like to improve.

📜 License

This project is licensed under the MIT License.

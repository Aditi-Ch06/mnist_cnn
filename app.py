import os
import torch
import torch.nn as nn
from flask import Flask, render_template, request
from PIL import Image
from torchvision import transforms

app = Flask(__name__)
app.config['ENV'] = os.environ.get('FLASK_ENV', 'production')

# Move model loading inside a try-except block
try:
    # Define the same CNN architecture
    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            # convolutional layers
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding="same")
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding='same')

            # fully connected layers
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(64*7*7, 1024)
            self.fc2 = nn.Linear(1024, 10)

            # activation function
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.relu(self.conv1(x))
            x = self.pool(x)
            x = self.relu(self.conv2(x))
            x = self.pool(x)
            x = self.flatten(x)
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    # Load the model
    model = CNN()
    model_path = os.path.join(os.path.dirname(__file__), 'mnist_cnn.pth')
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None

# Transform for input images
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

@app.route('/', methods=['GET', 'POST'])
def home():
    if model is None:
        return "Model failed to load", 500
        
    prediction = None
    if request.method == 'POST':
        # Get the image from post request
        file = request.files['file']
        img = Image.open(file).convert('L')  # Convert to grayscale
        
        # Transform image
        img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
        
        # Make prediction
        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted = torch.max(outputs, 1)
            prediction = predicted.item()
    
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    app.run(host='0.0.0.0', port=port) 
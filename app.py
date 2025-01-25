from flask import Flask, request, render_template
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import base64
from io import BytesIO
import os

app = Flask(__name__)

class_labels = {
    0: 'burger',
    1: 'butter_naan',
    2: 'chai',
    3: 'chapati',
    4: 'chole_bhature',
    5: 'dal_makhani',
    6: 'dhokla',
    7: 'fried_rice',
    8: 'idli',
    9: 'jalebi',
    10: 'kaathi_rolls',
    11: 'kadai_paneer',
    12: 'kulfi',
    13: 'masala_dosa',
    14: 'momos',
    15: 'paani_puri',
    16: 'pakode',
    17: 'pav_bhaji',
    18: 'pizza',
    19: 'samosa'
}

class FoodCNN(nn.Module):
    def __init__(self, num_classes=20):
        super(FoodCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FoodCNN(num_classes=20).to(device)
model.load_state_dict(torch.load('food_cnn_model_0.00005_29_epochs.pth', map_location=device))
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

def predict_image(image_file):
    if isinstance(image_file, Image.Image):
        image = image_file
    else:
        image = Image.open(image_file)
    
    image = image.convert("RGB")
    
    image = preprocess(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(image.to(device))
        _, predicted = torch.max(output, 1)
        food_name = class_labels[predicted.item()]
        return food_name

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'cropped_image' in request.form:
            cropped_image_data = request.form['cropped_image']
            cropped_image_data = cropped_image_data.split(',')[1]  # Remove the data URL prefix
            image_data = BytesIO(base64.b64decode(cropped_image_data))
            image = Image.open(image_data)
            
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            
            image_path = 'static/uploads/cropped_image.jpg'
            image.save(image_path, format='JPEG')

        elif 'image' in request.files:
            image = request.files['image']
            image_data = BytesIO(image.read())
            image = Image.open(image_data)
            
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            
            image_path = 'static/uploads/uploaded_image.jpg'
            image.save(image_path, format='JPEG')

        label = predict_image(image)

        return render_template('result.html', label=label, image_path=image_path)

    return render_template('index.html')

if __name__ == '__main__':
    app.run()

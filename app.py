
# import os
# from flask import Flask, render_template, request
# import numpy as np
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.models import load_model
# import torch.nn as nn


# app = Flask(__name__, static_folder="images")
# APP_ROOT = os.path.dirname(os.path.abspath(__file__))
# saved_model = load_model("vgg16_brain_model.h5", compile=False)  # Load your VGG model here

# class ConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(ConvBlock, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         x = self.conv(x)
#         return x

# class UpConv(nn.Module):
#     def __init__(self, in_channels, out_channels, bilinear=True):
#         super(UpConv, self).__init__()

#         if bilinear:
#             self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         else:
#             self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

#         self.conv = ConvBlock(in_channels, out_channels)

#     def forward(self, x1, x2):
#         x1 = self.up(x1)
#         x = torch.cat([x2, x1], dim=1)
#         x = self.conv(x)
#         return x

# class AttentionGate(nn.Module):
#     def __init__(self, in_channels):
#         super(AttentionGate, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
#             nn.BatchNorm2d(in_channels // 2),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels // 2, 1, kernel_size=1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         g = F.avg_pool2d(x, kernel_size=x.size()[2:])
#         g = self.conv(g)
#         x = x * g
#         return x

# class ResUNet(nn.Module):
#     def __init__(self, in_channels=3, out_channels=1, bilinear=True):
#         super(ResUNet, self).__init__()
#         self.bilinear = bilinear

#         self.down1 = ConvBlock(in_channels, 64)
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.down2 = ConvBlock(64, 128)
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.down3 = ConvBlock(128, 256)
#         self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.down4 = ConvBlock(256, 512)
#         self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

#         self.center = ConvBlock(512, 1024)

#         self.up4 = UpConv(1024, 512, bilinear)
#         self.attention4 = AttentionGate(512)
#         self.up3 = UpConv(512, 256, bilinear)
#         self.attention3 = AttentionGate(256)
#         self.up2 = UpConv(256, 128, bilinear)
#         self.attention2 = AttentionGate(128)
#         self.up1 = UpConv(128, 64, bilinear)
#         self.attention1 = AttentionGate(64)

#         self.final = nn.Conv2d(64, out_channels, kernel_size=1)  # Adjust based on your output requirements

#     def forward(self, x):
#         x1 = self.down1(x)
#         x2 = self.pool1(x1)

#         x3 = self.down2(x2)
#         x4 = self.pool2(x3)

#         x5 = self.down3(x4)
#         x6 = self.pool3(x5)

#         x7 = self.down4(x6)
#         x8 = self.pool4(x7)

#         # Center
#         x_center = self.center(x8)

#         # Upsample
#         x_up4 = self.up4(x_center, x7)
#         x_att4 = self.attention4(x_up4)

#         x_up3 = self.up3(x_att4, x6)
#         x_att3 = self.attention3(x_up3)

#         x_up2 = self.up2(x_att3, x5)
#         x_att2 = self.attention2(x_up2)

#         x_up1 = self.up1(x_att2, x4)
#         x_att1 = self.attention1(x_up1)

#         # Final layer
#         x_final = self.final(x_att1)

#         return x_final

# def check(input_img_path):
#     print("Your image is: " + input_img_path)

#     # Load and preprocess the image using tf.keras
#     img = image.load_img(input_img_path, target_size=(224, 224))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array /= 255.0  # Normalize pixel values to [0, 1]

#     # Make predictions using the loaded model
#     predictions = saved_model.predict(img_array)
#     print("Raw Predictions:", predictions)

#     # Interpret the predictions (assuming a binary classification)
#     tumor_probability = predictions[0][0]

#     # Adjust the threshold based on your model and data characteristics
#     threshold = 0.5

#     # Use numeric labels (0 and 1) based on the binary classification setup
#     predicted_class = 1 if tumor_probability >= threshold else 0

#     if predicted_class == 1:
#         status = "Brain Tumor Detected"
#     else:
#         status = "No Brain Tumor Detected"

#     print("Tumor Probability:", tumor_probability)
#     print("Predicted Class:", predicted_class)
#     print("Final Status:", status)

#     return status


# @app.route('/')
# @app.route('/index')
# def index():
#     return render_template('upload.html')

# @app.route('/upload', methods=['GET', 'POST'])
# def upload():
#     target = os.path.join(APP_ROOT, 'images/')
#     print(target)

#     if not os.path.isdir(target):
#         os.mkdir(target)

#     for file in request.files.getlist('file'):
#         print(file)
#         filename = file.filename
#         print(filename)
#         dest = '/'.join([target, filename])
#         print(dest)
#         file.save(dest)

#         # Call the check function to get the prediction status
#         status = check(dest)
#         print("Status before rendering template:", status)
        
#     return render_template('complete.html', image_name=filename, predvalue=status)

# if __name__ == "__main__":
#     app.run(port=4555, debug=True)

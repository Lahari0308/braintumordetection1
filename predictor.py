# import numpy as np
# from keras.preprocessing import image
# from tensorflow.keras.models import load_model
# saved_model = load_model("model/VGG_model.h5")
# status = True


# def check(input_img):
#     print(" your image is : " + input_img)
#     print(input_img)

#     img = image.load_img("images/" + input_img, target_size=(224, 224))
#     img = np.asarray(img)
#     print(img)

#     img = np.expand_dims(img, axis=0)

#     print(img)
#     output = saved_model.predict(img)

#     print(output)
#     if output[0][0] == 1:
#         status = True
#     else:
#         status = False

#     print(status)
#     return status

import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Load the pre-trained VGG model
saved_model = load_model("model/VGG_model.h5")

def check(input_img_path):
    print("Your image is: " + input_img_path)

    # Load and preprocess the image using tf.keras
    img = image.load_img(input_img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize pixel values to [0, 1]

    # Make predictions using the loaded model
    predictions = saved_model.predict(img_array)
    print("Raw Predictions:", predictions)

    # Interpret the predictions (adjust this part based on your model output)
    tumor_probability = predictions[0][0]

    # Adjust the threshold based on your model and data characteristics
    threshold = 0.5

    if tumor_probability >= threshold:
        status = True
    else:
        status = False

    print("Tumor Probability:", tumor_probability)
    print("Final Status:", status)

    return status

# Example usage:
# Replace 'path_to_your_image.jpg' with the actual path to your image
image_path = 'path_to_your_image.jpg'
check_result = check(image_path)
print("Check Result:", check_result)

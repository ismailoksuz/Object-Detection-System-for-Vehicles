import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np

class_names = ['Ambulance', 'Bicycle', 'Bus', 'Car',
               'Motorcycle', 'Taxi', 'Truck', 'Van']
img_height = 180
img_width = 180
# Load the saved model
model = keras.models.load_model('vehicle_model')

# Download and preprocess the image
image_url = 'https://lh3.googleusercontent.com/fife/AAbDypDzcIoW3ojH-6KRV7JL2zfjB38ZTxq0aV5Zkl3ImX1a9PRuocl0Dv6ZPjP3ytLRYIRrmh-4myYfRiK55SNSaS3oUMUJENBoQDkbslHVk7EvPzpffLJIghoDuPA2YgLTFzw9-8GEKcbMmzkUHPPMHpD1Z-eCJyLxi33Tr7_jpd6MgTWq7nEgiXMbC4PdliC-TxRKraa7BNr8FgYUuwmWJVaWgcUSBoGymllOk_g4zSM_fZokVqm99QmsGuzqUxXf-SSVdeUxZnoGo24_kvRZtczn04oq3-BmhHgHj4HFo34XjmJjfkZEIoeh197wc8To6CPZgaLOjEo9uPclSyvG0jRdNqLOtXJ3paDfKkIH-a21tlP8eUUTsAheFFaI0TBWZ4P2GFHfahpsoq7flULZK6zBbPARe8rEfiu5dotepix0ZOqnxYisQH-Axow0kedC-BaANd0YzeSfPKfle33-7z0tUCcWrmMwVePn060FAYy84mr00fdvj761ZikzJsoy2mpE-dYqQYzZUKnRiS2fydEUUmT3JQ2QX1V0ifu8sYhOrrdyxffIPwLSkQ_ElxduNNt3Ba6bpzw0k9vUdosG6K4i6_gyPtidUzNFJP9OX6xRNokHQGWkuNJDKcYoZ2g1YZ8gaYVslbRtPfbMSaScg8QV5fZtWOpGXZO-C_R20sVbNNUtvUJGrVNA3yl5e-tO1akqnfyejLGsX27Q_8gSm2TnY_afc3eRt2ze0DlKMzraH1Yq3NXMNvnH6gXRT8RceWtZ5xNcJIkKECIXhpf_GvODSDDIgx-fhHQbbXLU-G5OykDNEu-RwNetAK2hVj55Q7tj3EOSulqaFnB9wMysePEiiacpgND1fB_bsP5K6iXHwzqxL_nZL2UW5ccrX6GlffkjA7PCTju5j3joqdGQgXOiVZNOy9LwCCrWOp6jq1yJOyK35R4ZdIf2yKt0FHpS8qz4LUCxw6Brwj0Kc9RubAgkpBUGgnE4HC0QiHiAGEE9fxsROkP9DiY4TIpbwy90C9zwTvgN0wG8o824yHwBhChwXFoubYL87bEFEXyxbl_Iuu1ERrEn3Mj16urc1q9CwVREESCNKTqmLAprf7v6QbzzFRgn28zhvgsqhi1n60KKfNsurWZb0ieRfWAbGP0lvCv_uTUZDyREKADnUnZMdfWC0Y8DvbjoopstAf8kn8kx8Q15E-ZtLhI_DphKcvxNIFGz4YE1bjCqbrFJHb-Y2pEdbiihxnBVtgStm6UTeMEl99rwqSCb3E4f_uArgZZUvaoTCb1LkcetYVwTMfITDjcp02irjGFrZABZbyXZkxORPalmD4dZrBgwSwa9r0ek1O2cGK0-Yb00XJT2qBqjEbKSMjOYc0ZWqF7cXfJ-GgCRpOQNzoDqK0HNUYQC1B-PmSridMcZ5jp_fj8zWJFHmqTH-zq52kbTfzdbn_CTHHXQne6Tm4Y0JNrO_B-zL42gMTNByrhy2ohr4xbTtFgVDQ=w1920-h878'
image_path = tf.keras.utils.get_file('arrraba', origin=image_url)

original_img = tf.keras.utils.load_img(image_path)
img = tf.keras.utils.load_img(
    image_path, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create a batch

predictions = model.predict(img_array)
scores = tf.nn.softmax(predictions[0])

# Get the top three indices
top_indices = tf.math.top_k(scores, k=3).indices

# Plot the image
plt.imshow(original_img)
plt.axis('off')

# Print the top three guesses with confidence levels
print("Top 3 guesses:")
for i in range(3):
    print("{}: {:.2f}%".format(class_names[top_indices[i]], 100 * scores[top_indices[i]]))

# Add the top three guesses and confidence levels to the plot
text = "Top 3 Guesses:\n"
for i in range(3):
    text += "{}: {:.2f}%\n".format(class_names[top_indices[i]], 100 * scores[top_indices[i]])
plt.text(0, -20, text, fontsize=14, ha="left", va="top", bbox=dict(facecolor="w", alpha=0.5))

plt.show()

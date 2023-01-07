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
image_url = 'https://mcn-images.bauersecure.com/wp-images/58950/615x405/bmw_gs310r.jpg'
image_path = tf.keras.utils.get_file('example44', origin=image_url)

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

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

class_names = ['Ambulance', 'Bicycle', 'Bus', 'Car',
               'Helicopter', 'Limousine', 'Motorcycle', 'Taxi', 'Truck', 'Van']
img_height = 180
img_width = 180
# Load the saved model
model = keras.models.load_model('vehicle_model')

# Download and preprocess the image
sunflower_url = 'https://contents.mediadecathlon.com/s922848/k$48172fb53121a8d65b84ec23ad3fe28b/btwin%20velo%2016%20%20500%20dark%20hero.jpg'
sunflower_path = tf.keras.utils.get_file('yeni', origin=sunflower_url)

img = tf.keras.utils.load_img(
    sunflower_path, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

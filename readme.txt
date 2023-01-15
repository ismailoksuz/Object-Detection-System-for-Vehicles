In order to run this code, you should first have the libraries like matplotlib, keras etc. installed
To install the libraries, use "pip install keras", "pip install matplotlib etc".

Then, you should run first the main.py file to train and save the model. It will save the model to a folder called
vehicle_model. If vehicle_model is already existing, you should delete it before running main.py file. To, run it, write "python main.py" to your terminal. And you need to have the dataset folder in the same directory as well. Since it takes so much time, we will not upload the dataset. But you can find its link in our report.

After you run the main.py file, you will have the model in the folder vehicle_model. To test this model, you should run
test.py file by typing "python test.py" to the terminal. When you run that file, it will load the saved vehicle_model, and
asks for an image url to test. You should provide it with an image url you find from the web.
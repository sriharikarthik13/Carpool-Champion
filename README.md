# Welcome to the Traffic Prediction and Carpool Effect Study using Deep Learning Project.


-This project was built in the ambition of trying to convey the positive effects that the adoption of carpooling can bring to heavily traffic congested cities like Bangalore.

- This project takes in the data of number of cars crossing a particular point at a particular time of the day.

- This project has two main components, one it predicts the number of cars passing a particular point a peak hours like 11:00 AM and 8:00 PM by training on input data, which contains the number of cars at the time intervals of:=> 7:00 AM, 8:00 AM, 9:00 AM and 10:00 AM for predicting the flow at 11:00 AM, and similarly the car count at 4:00 PM, 5:00PM, 6:00PM and 7:00PM to predict the traffic flow at 8:00PM.

- The morning set of data is collected by using an ANPR model that is build using deep learning(KERAS), where a set of car images which are or can be retreived from a inplace camera and the car plate text is extracted from each car and stored in a dataframe.

- The evening set of data is collected by using the ultraalytics feature of the YOLO library to detect the objects/cars present in a given video, we provide 4 such different videos for the 4 input hours and builts the dataframe for input.

- A LSTM Model which utilises RNN to built a deep learning model which takes in the sequential input length of 3, that is 3 inputs to predict one or the next output.

- Along with this we also incoporate the concept of carpooling by utilizing a small algorithm which demonstrates the effect of carpooling and the percentage decrease it brings in car count and time taken for a car to go from one point to another.

"# Carpool-Champion" 
"# Carpool-Champion" 
# Carpool-Champion

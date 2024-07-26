#Importing the required libraries
import os
import numpy as np
from datetime import datetime, timedelta

from werkzeug.datastructures import accept
#from anpr import anpr_extract
import pandas as pd
import cv2
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt



#Creating the dataframe, which contains the data for number of cars crossing a point A for the hours of - 7,8,9 and 10
#Converting it as a function
def create_dataframe(image_folder):
    count_data=[]
    for image_file in os.listdir(image_folder):
        if image_file.endswith('.jpg'):
            image_path = os.path.join(image_folder,image_file)
            a,b,plate_text = anpr_extract(image_path)
            if plate_text:
                timestamp_str = image_file.split('_')[1].split('.')[0]
                timestamp = datetime.strptime(timestamp_str,"%Y%m%d%H%M%S")
                count_data.append([timestamp,plate_text])
#Creating a DataFrame from the attained data
    df = pd.DataFrame(count_data,columns=['timestamp','plate_text'])
    df.set_index('timestamp', inplace=True)

    #Now reshaping the created DF into such a manner that it lists out the car count for each hour.
    df['count']=1
    df_resampled = df.resample('H').sum()
    return df_resampled

#df_resampled = create_datafrma('IMAGE_FILES')
#Utilizing the created csv file that stores the dataframe created by using the above, 
df_resampled = pd.read_csv('car_count_data.csv')

#Next, we are going to build the LSTM Model used to predict the car count at 11 oclock.
#We develop the predictive model by creating a Sequential model to which we apply the LSTM(Long Short-Term Memory) which is an RNN used for prediction.
#We use 50 units/cells in each layer of the model to learn the pattern of the data.
#The model has an input of one, which is the car_count. In the later we have an input of 2 -> car_count and time_taken
#Create a Dense layer with an output of 1 Unit neuron in the layer. Output -> car_count(Takes output from the lSTM layers)
#We fit the model to input an output(train) data through 200 epochs, the number of times the model is run to fit the data.
#Also ensuring we dont overfit the model.
#Creating a funtion to output the predicted value.
#Also introducing the concept of carpooling, where the user decides the number of people opting for carpooling,, this will use a basic algorithm to predict the number of cars if carpooling is undertaken.
#Also outputs the graph, which compares no carpool predicted value to carpool implemented predicted value.
#Creating all of this as a function
#Importing the required libraries
def carpool_prediction(carpool_amount):

#Let us start by normalizing the input values for the LSTM Model by MinMaxScaler
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()

#Creating a new column for the normalised values
    df_resampled['norm_count'] = scaler.fit_transform(df_resampled[['count']])

#Next step is to create sequences to feed into the model, we achieve this by creating a function create_sequences.
#This function takes in the data and the sequence length: In this case the sequence length is 3, as our input size is 3 and output is 1.
#Point to note here is that, we can also implement a similar model to take in car counts of 3 peak days say Monday,Tuesday, and Wednesday for certain period and use the model to predict the value for count of cars on a thursday.\
#Create the X and y values to input into the model/train the model.

    def create_sequences(data, seq_length): 
        X = [] 
        y = [] 
        for i in range(len(data) - seq_length): 
            X.append(data[i:i + seq_length]) 
            y.append(data[i + seq_length]) 
        return np.array(X), np.array(y) 
# Define the sequence length 
    seq_length = 3
#Defining X and y
    X, y = create_sequences(df_resampled['norm_count'].values, seq_length)  

#Now we create the lSTM Model
    model = Sequential()
#We specify 1 as the input_shape as this model only takes one input, as mentioned previously
    model.add(LSTM(50,activation='relu',input_shape=(seq_length,1)))
#Adding Dense 1 for the single output
    model.add(Dense(1))
#Adding optimizer and loss for the model
    model.compile(optimizer='adam',loss='mse')

#Now we need to train the model with 200 epochs
    model.fit(X,y,epochs=200,verbose=1)

#Now we need to create the last sequence that is the sequence we want the model to predict and reshape it:
    predict_sequence = df_resampled['norm_count'].values[-seq_length:].reshape((1,seq_length,1))

#Now to predict the value at 11:
    predicted_value = model.predict(predict_sequence)

#Using inverse transform to get the actual car_count from the normalised value.
    actual_predicted_value = scaler.inverse_transform(predicted_value)[0][0]

#Now let us create the basic algorithm for including the concept of carpooling into this car_count prediction:
# User input for carpool rate at 11 o'clock let us say for example: 20 people opt for carpooling

# Adjusting predicted car count based on carpool rate
    number_of_people_accepting_carpool = carpool_amount
    val = actual_predicted_value - number_of_people_accepting_carpool
    val_2 = number_of_people_accepting_carpool//4
    val_3 = number_of_people_accepting_carpool%4

    adjusted_predicted_car_count = val + val_2 + val_3

#Now let us build the plot to showcase the effect of carpooling on the predicted car count:

    original_carcounts = []
    original_times = ['7AM', '8AM', '9AM', '10AM']
    for i in df_resampled['count']:
        original_carcounts.append(i)

# Adding the predicted car count for 11 o'clock
    original_times.append('11AM')
    original_carcounts.append(actual_predicted_value)



#Creating a list for the updated car counts:
    adjusted_car_counts = original_carcounts[:-1]
    adjusted_car_counts.append(adjusted_predicted_car_count)

    percentage_count = ((actual_predicted_value - adjusted_predicted_car_count)/actual_predicted_value)*100
   

#Plotting
    plt.figure(figsize=(12,6))
    plt.plot(original_times,original_carcounts,marker='o',label='Original Car Count')
    plt.plot(original_times,adjusted_car_counts,marker='X',label='Car Count after Carpooling Implementation')
    plt.title("Car Count Prediction and Impact of Carpooling")
    plt.xlabel('Time of Day')
    plt.ylabel('Car Count')
    plt.legend()
    plt.grid(True)
    plt.savefig('./static/Carpool_Impact.png')
    return  percentage_count,actual_predicted_value,adjusted_predicted_car_count


#predicted,carpool_predicted = carpool_prediction(20)    


##############CODE FOR DETECTING THE CARS IN A VIDEO USING YOLO#############################################
#Now we will utilize YOLOv8 to detect the cars crossing a particular point, and build our model for predicting the value at 8, using the same LSTM model.
#First we import the required ultralytics library.
#We also import supervision which is used to count the detected cars
#We also utilize BOT-SORT for the detection/tracking of the cars crossing a particular point.
def detect_cars(input_video):
    import ultralytics
    import torch
    import supervision as sv
    from ultralytics import YOLO
    import cv2
    from collections import defaultdict

# Load the pretrained YOLOv8n model
    model = YOLO('yolov8n.pt')

#Predict or implement the yolo model on your video:
    model.predict(source=input_video, save=True, imgsz=320, conf=0.5)

#Now we utilize supervision and YOLO, to detect the number of cars crossing a point, we also append these findings to a newly created video by creating a video sink.
# Set up variable to start capturing the intended video
    capture = cv2.VideoCapture(input_video)

#Creating a dictionary to store the number of cars crossing
    all_objects={}

#Importing the input video
    video_info = sv.VideoInfo.from_video_path(input_video)
#Creating the video sink to analyze the objects in the input video:
    with sv.VideoSink("output.mp4", video_info) as sink:
    #Read each frame of the input video
        while capture.isOpened():
            success, frame = capture.read()

            if success:
            # Run YOLOv8 tracking on each of the frames, to check for objects : 2-> represent cars
                results = model.track(frame, classes=[2], persist=True, save=True, tracker="bytetrack.yaml")

            # Track the boxes and track_ids of the detected cars
                boxes = results[0].boxes.xywh.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()
            #Count the number of boxes being detected in the video.
                for i in track_ids:
                    all_objects[i] = True
            #Store the number of cars detected:
                car_count = len(all_objects)
            
            # Plot the results on the frame and append it to the output video
                annotated_frame = results[0].plot()
                sink.write_frame(annotated_frame)
            else:
                break

# Release the video capture
    capture.release()
    return car_count
        
    

#Building a function to run the YOLO detection model on different videos to count the number of cars in each video.
def create_data(n):
    car_count_data=[]
    for i in range(n):
        strimage = f"video_{i+1}.mp4"
        car_count = detect_cars(strimage)
        car_count_data.append(car_count)
    return(car_count_data)



##########CODE FOR TIME EFFECT ON CARPOOLING ####################

def carpool_time_effect(car_pool_acceptance):
    #car_count_data = create_data(4)
    car_count_data=[24,24,74,67]

    time_data=[]
    def time_calculator(car_data,time_data):
        for i in car_data:
            if i > 60:
                time_data.append(40+i%10)
            elif i<60 and i>30:
                time_data.append(30 + i%10)
            else:
                time_data.append(30)
        return time_data
    time_data = time_calculator(car_count_data,time_data)

#For 4oclock, 5oclock, 6oclock and 7oclock
    time=[4,5,6,7]
#Now let us move to the final part of this analysis, where will build a lstm model to predict the car count at a particular time along with the time taken to cross from one point A to another point B
#This process, involves the same steps as that of the previous model we built the only difference is here we will have 2 inputs and 2 ouputs from the model
#First let us create the data for the model:
    data_val=[]
    for i in range(4):
        data_val.append([car_count_data[i],time_data[i]])

    df = pd.DataFrame(data_val,columns=['car_count','time_taken'],index=time)

#Let us start by normalizing the input values for the LSTM Model by MinMaxScaler
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()

#Creating a new column for the normalised values
    scaled_data = scaler.fit_transform(df[['car_count','time_taken']])

#Next step is to create sequences to feed into the model, we achieve this by creating a function create_sequences.
#This function takes in the data and the sequence length: In this case the sequence length is 3, as our input size is 3 and output is 1.
#Point to note here is that, we can also implement a similar model to take in car counts of 3 peak days say Monday,Tuesday, and Wednesday for certain period and use the model to predict the value for count of cars on a thursday.\
#Create the X and y values to input into the model/train the model.

    def create_sequences(data, seq_length): 
        X = [] 
        y = [] 
        for i in range(len(data) - seq_length): 
            X.append(data[i:i + seq_length]) 
            y.append(data[i + seq_length]) 
        return np.array(X), np.array(y) 
# Define the sequence length 
    seq_length = 3
#Defining X and y
    X, y = create_sequences(scaled_data, seq_length)  

#Now we create the lSTM Model
    model = Sequential()
#We specify 2 as the input_shape as this model only takes two input, as mentioned that is -> car_count and time_taken
    model.add(LSTM(50, activation='relu', input_shape=(seq_length, 2)))
#Adding Dense 2 for the two outputs
    model.add(Dense(2))
#Adding optimizer and loss for the model
    model.compile(optimizer='adam',loss='mse')

#Now we need to train the model with 200 epochs
    model.fit(X,y,epochs=200,verbose=1)

#Now we need to create the last sequence that is the sequence we want the model to predict and reshape it:
    predict_sequence = scaled_data[-seq_length:]
    predict_sequence = np.expand_dims(predict_sequence, axis=0)

#Now to predict the value at 11:
    predicted_value = model.predict(predict_sequence)

#Using inverse transform to get the actual car_count from the normalised value.
    actual_predicted_count,actual_predicted_time = scaler.inverse_transform(predicted_value)[0]


#Now let us create the basic algorithm for including the concept of carpooling into this car_count prediction:
# User input for carpool rate at 8 o'clock let us say for example: 10 people opt for carpooling

# Adjusting predicted car count based on carpool rate

    number_of_people_accepting_carpool = car_pool_acceptance
    val = actual_predicted_count - number_of_people_accepting_carpool
    val_2 = number_of_people_accepting_carpool//4
    val_3 = number_of_people_accepting_carpool%4

    adjusted_predicted_car_count = val + val_2 + val_3


#Now let us find the time it takes for a car to cross point A to B after implementing car pooling:
    time_data_carpool = time_calculator([adjusted_predicted_car_count],[])
    time_data_carpool = time_data_carpool[0]


#Now let us build the list of predicted values before and after carpooling and create the graphs to visualize
    labels = ['Before Carpooling', 'After Carpooling']
    car_counts = [actual_predicted_count, adjusted_predicted_car_count]
    times_taken = [actual_predicted_time, time_data_carpool]

    percentage_count = ((actual_predicted_count - adjusted_predicted_car_count)/actual_predicted_count)*100
    percentage_time = ((actual_predicted_time - time_data_carpool)/actual_predicted_time)*100
    print(f"Percentage decrease in Car count after Car Pooling is : {percentage_count}")
    print(f"Percentage decrease in Car count after Car Pooling is : {percentage_time}")

    plt.figure(figsize=(14, 6))
# Car Count
    plt.subplot(1, 2, 1)
    plt.bar(labels, car_counts, color=['orange', 'blue'])
    plt.title('Predicted Car Count at 8 oclock')
    plt.xlabel('Carpooling Scenario')
    plt.ylabel('Car Count')
    plt.ylim(0, max(car_counts) * 1.2)
# Time Taken
    plt.subplot(1, 2, 2)
    plt.bar(labels, times_taken, color=['orange', 'blue'])
    plt.title('Predicted Time Taken at 8 oclock')
    plt.xlabel('Carpooling Scenario')
    plt.ylabel('Time Taken (minutes)')
    plt.ylim(0, max(times_taken) * 1.2)
    plt.tight_layout()
    plt.savefig('./static/Carpool_Impact_OnTime.png')
    return percentage_count,percentage_time,actual_predicted_count,adjusted_predicted_car_count,actual_predicted_time,time_data_carpool


# test script to parse screenshots captured from stream to determine which ones are worth uploading for further analysis

import csv
import json
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import pyimgur
import pyautogui
from time import sleep
import RepoUpdate
from datetime import datetime
import matplotlib.pyplot as plt
import os
from picamera import PiCamera

# variable that determines if images are captured from screenshots, or the camera directly
image_capture = "SCREENSHOT" # set to "SCREENSHOT" to take screenshots

if (image_capture == "CAMERA"):
    camera = PiCamera()

# An ordered list of the different possible outputs of the model in the order that they appear in the output array
outputs = ["person", "no_person"]

# imgur client id
CLIENT_ID = "8592d136b96e6c6"

# disable scientific notation for clarity
np.set_printoptions(suppress=True)

# load the model
print("loading model...")
model = tf.keras.models.load_model("/home/pi/Desktop/FishLadderStreamAnalyzer/keras_model.h5")
print("model loaded!")


# create the array of the right shape to feed into the keras model
# the 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

def run_model(up_image):
    sleep(0.2)
    print("looking for face...")

    # loading the image
    image = up_image.convert('RGB')
    print("opened image...")

    # resize the image to a 224x224:
    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # load the image into the array
    data[0] = normalized_image_array

    # run the inference
    print("running model...")
    prediction = model.predict(data)
    #print("ran model: " + str(prediction))
    
    # printing the confidence values
    # turning the numpy data array into a list
    prediction_string = str(prediction)
    split_str = prediction_string.split()
    split_str.pop()

    # making newlines
    print("\n\n")

    # turning the list into a dict with keys of the animal names and values of the confidence values
    res_str = dict(zip(outputs, split_str))
    # remove unwanted characters in the bear string
    res_str["person"] = str(res_str["person"]).replace('[', '')
    print(res_str)

    # getting the max confidence value out of the dictionary
    max_value = prediction[0][0]

    # checking uf the program is more then 50% sure
    if (float(max_value) > float(0.5)):
        print("fish detected!")

        # print a newline
        print("")
        # setting a variable to represnt the path to the image
        PATH = "screenshot.png"
        # connecting to imgur
        im = pyimgur.Imgur(CLIENT_ID)

        # uploading the image, the variable is the imgur link
        uploaded_image = im.upload_image(PATH, title="Uploaded with PyImgur")

        # printing the imgur link 
        print(uploaded_image.link)

        # adding a list to put in the csv
        changes = [[res_str, uploaded_image.link, datetime.now(), "1"]] # new row, including the model's confidence in its decision, link to the image, timestamp, and a blank column that we will use to count the number of fish at some point

        # opening the data json in append mode
        with open(r"./FishLadderStreamCapture/convertcsv.csv","a") as f:                 
            #initalizing the csv writer                   
            writer = csv.writer(f) # writing the new row from the changes list                                        
            writer.writerows(changes)
            f.close()

            # update the graph
            csv_file = csv.reader(open("./FishLadderStreamCapture/convertcsv.csv"))

            times = []

            for row in csv_file:
                try:
                    times.append(row[2].split(" ")[0])
                except:
                    pass

            # count the number of fish for each day
            frequencies = dict()
            for date in times:
                if not (date in frequencies.keys()): # new date
                    frequencies[date] = 1
                else:
                    frequencies[date] += 1
                
            #print(frequencies.keys())
                    
            # update the json file with the count data (used by the graph generated on the website)
            j = {"lables": list(frequencies.keys()), "data": list(frequencies.values())}
            jsn = open(r"./FishLadderStreamCapture/convertjson.txt", "w") # write changes to json
            json.dump(j, jsn)
            jsn.close()

            ax = plt.subplot(111)
            plt.xticks(rotation=90)
            plt.xlabel("Date")
            plt.ylabel("Fish counted")
            plt.title("Fish Counted By Day")
            plt.grid(True)

            values = []
            for value in frequencies.values():
                values.append(value)
            l = plt.fill_between(frequencies.keys(), values)

            plt.gcf().autofmt_xdate()
            plt.plot(frequencies.keys(), values)

            ax.set_xlim(0, len(frequencies.keys())-1)
            ax.set_ylim(0, max(values)+int(max(values)/5))

            l.set_facecolors([[.5,.5,.8,.3]])

            # change the edge color (bluish and transparentish) and thickness
            l.set_edgecolors([[0, 0, .5, .3]])
            l.set_linewidths([3])

            # add more ticks
            ax.set_xticks(np.arange(len((frequencies.keys()))))

            # remove tick marks
            ax.xaxis.set_tick_params(size=0)
            ax.yaxis.set_tick_params(size=0)

            # change the color of the top and right spines to opaque gray
            ax.spines['right'].set_color((.8,.8,.8))
            ax.spines['top'].set_color((.8,.8,.8))

            # tweak the axis labels
            xlab = ax.xaxis.get_label()
            ylab = ax.yaxis.get_label()

            xlab.set_style('italic')
            xlab.set_size(10)
            ylab.set_style('italic')
            ylab.set_size(10)

            # tweak the title
            ttl = ax.title
            ttl.set_weight('bold')

            try: # delete the file if it exists
                os.remove("./FishLadderStreamCapture/graph.png")
            except:
                pass
            plt.savefig("./FishLadderStreamCapture/graph.png")
            ax.clear()

            # push the updated repo containing the csv and graph to github
            RepoUpdate.git_push()

try:
    while True:
        if (image_capture == "CAMERA"):
            im = camera.capture("screenshot.png")
        elif (image_capture == "SCREENSHOT"):
            im = pyautogui.screenshot("screenshot.png")

        run_model(Image.open("screenshot.png"))
except KeyboardInterrupt:
    pass

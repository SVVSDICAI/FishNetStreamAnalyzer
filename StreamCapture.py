# test script to parse screenshots captured from stream to determine which ones are worth uploading for further analysis

import csv
import cv2
import json
import tensorflow as tf
from PIL import Image, ImageFile, ImageOps
import numpy as np
import pyimgur
from colorama import Fore, Style
from time import sleep
import RepoUpdate
from datetime import datetime
import matplotlib.pyplot as plt
import os

ImageFile.LOAD_TRUNCATED_IMAGES=True

SHOW_PREDICTION_IMGS = True # this boolean determines if the script will display the model's predictions as images

# An ordered list of the different possible outputs of the model in the order that they appear in the output array
outputs = ["fish", "no_fish"]
species_classes = ["White Sucker", "Black Bullhead Catfish", "Plains Topminnow", "Brown Trout", "Creek Chub"]

# imgur client id
CLIENT_ID = os.getenv("IMGUR_CLIENT_ID") # TODO set this on the device running this script
print('--------------------------------------------\n' + CLIENT_ID)
# disable scientific notation for clarity
np.set_printoptions(suppress=True)

# load the models
print("loading unary model...")
# the unary model is the primary model that determines if there is a fish in the image or not
# it is called unary because in effect it just counts up the number of frames with fish
unary_model = tf.keras.models.load_model("./models/unary_classifier/keras_model.h5")
print("unary model loaded!")

print("loading species classifier model...")
species_model = tf.keras.models.load_model("./models/species_classifier/keras_model.h5")
print("species classifier model loaded!")

# create the array of the right shape to feed into the keras model
# the 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# the model should only count fish when it sees a new fish
# we want to avoid counting the same fish multiple times, so only increment the counter if there were no fish in the previous frame
# boolean to keep track of if there was any fish in the previous frame
fish_prev_frame = False

clip_date = ''

def run_model(up_image):
    global fish_prev_frame

    sleep(0.2)
    print("looking for fish...")

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
    print("running unary model...")
    prediction = unary_model.__call__(data)
    print("ran unary_model: " + str(prediction))
    # making newlines
    print("\n\n")

    # checking if the program is more then 50% sure
    if (prediction[0][0] > float(0.5)):
        print("fish detected!")
        if SHOW_PREDICTION_IMGS:
          cv2.imshow("F  I  S  H  !  !  !", image_array)
          cv2.waitKey(1)

        # check to see if this is a new fish
        if not fish_prev_frame:
            # print a newline
            print("")
            
            # run through the species classifier:
            species_prediction = species_model.__call__(data)
            res_str = dict(zip(species_classes, species_prediction))
            print("ran species_model")
            
            species = dict()
            # match each species with the corresponding confidence and determine the most likely species
            predicted_fish = ['', 0]
            n = 0
            for fish_type in species_classes:
                species[fish_type] = float(species_prediction[0][n])
                if species[fish_type] > predicted_fish[1]:
                    predicted_fish = [fish_type, species[fish_type]]
                n += 1

            print(Fore.GREEN + "predicted species: " + predicted_fish[0])
            print(Style.RESET_ALL)

            # setting a variable to represent the path to the image
            PATH = "fish.png"

            # connecting to imgur
            im = pyimgur.Imgur(CLIENT_ID)

            # uploading the image, the variable is the imgur link
            uploaded_image = im.upload_image(PATH, title="Uploaded with PyImgur")

            # printing the imgur link 
            #print(uploaded_image.link)

            # adding a list to put in the csv
            fish_date = clip_date if clip_date != '' else datetime.now()
            changes = [[predicted_fish[0], uploaded_image.link, fish_date, "1"]] # new row, including the model's confidence in its decision, link to the image, timestamp, and a blank column that we will use to count the number of fish at some point

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
                    os.remove("./FishNetStreamCapture/graph.png")
                except:
                    pass
                plt.savefig("./FishNetStreamCapture/graph.png")
                ax.clear()

                # push the updated repo containing the csv and graph to github
                RepoUpdate.git_push()
                
                fish_prev_frame = True

    else:
        # no fish
        if SHOW_PREDICTION_IMGS:  
          fish_prev_frame = False
          cv2.imshow("NOOO F  I  S  H  !  !  !", image_array)
          cv2.waitKey(1)


if __name__ == '__main__':
    analysis = 'local_video' # set to 'live_stream' to analyze live stream

    try:
        if analysis == 'local_video':
            # iterate over each frame in the video file captured locally on the camera pi
            for clip in sorted(os.listdir('./clips/Deployment')):
                video = './clips/Deployment/' + clip
                cap = cv2.VideoCapture(video)
                clip_date = clip.split('_')[1].split('.')[0] # just get the date from file name of the form clipN_[date].h264

                # iterate over the frames
                while cap.isOpened():
                    # read the current frame
                    ret, frame = cap.read()

                    if not ret:
                        break

                    frame_image = Image.fromarray(frame)
                    frame_image.convert('RGB').save('fish.png')
                    # rotate the image
                    #frame_image = frame_image.rotate(90)
                    run_model(frame_image)

                # release the video capture object
                cap.release()

        elif analysis == 'live_stream':
            # The following is an example of how the above backend could be used to monitor a live stream running off of the camera Pi
            import pafy

            # set up pafy to capture images from the youtube stream
            url = '[put youtube url here]'
            video = pafy.new(url)
            best = video.getbest(preftype="mp4")

            capture = cv2.VideoCapture(best.url)
            while True:
                grabbed, frame = capture.read() # get the latest frame from the live stream
                run_model(frame)

    except KeyboardInterrupt:
        print('exiting...')

import numpy as np
import seaborn
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers, models
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn import svm
from sklearn.metrics import f1_score, accuracy_score
import seaborn as sns

def get_images_and_labels(file, name_folder, x): #this is the function to read data from folders
    training_images = []
    training_labels = []
    f = open(file, "r")
    lines = f.readlines()
    for line in lines[1:]: #from line 1 because the first line is "id,label"
        #if we read data from a file that contains images and labels, we use rstrip (to cut \n) and split the line in 2 components [name_of_image, image_label]
        #else if the file does not contain labels (we use the x variable to tell us if it does or not) and just read the name of images.
        line = line.rstrip("\n").split(",") if x == 1 else line.rstrip("\n")

        #if the file contains labels we open the image (name_folder + line[0] is the name of the image) using PIL library and transform that image into a np.array
        image = np.array(Image.open(f"./{name_folder}/{line[0]}")) if x == 1 else np.array(Image.open(f"./{name_folder}/{line}")) #array of pixels
        #append the image
        training_images.append(image)
        #if the file contains labels, we append the label as an int values
        if x == 1:
            training_labels.append(int(line[1]))
    f.close()

    if x == 0: #if the file does not contain labels, we return just the images
        return training_images
    #otherwise we return both, images and labels
    return training_images, training_labels

#MODEL 1

(training_images, training_labels) = get_images_and_labels("train.txt", "train+validation", 1) #1 -> cu label, 0 -> fara
(validation_images, validation_labels) = get_images_and_labels("validation.txt", "train+validation", 1)
test_images = get_images_and_labels("test.txt", "test", 0)

training_images = np.array(training_images)             #it was a simple list of np.arrays, now we transform it into an np.array of np.arrays (it's easier to work with them)
training_labels = np.array(training_labels)
validation_images = np.array(validation_images)
validation_labels = np.array(validation_labels)
test_images = np.array(test_images)

class_names = [0, 1, 2, 3, 4, 5, 6]

training_labels_one_hot = tf.keras.utils.to_categorical(training_labels)            #for a better and faster operation we transform the array of labels into a matrix with length of the vector as number of line and
                                                                                    #len(class_names) as the number of columns
                                                                                    #example class 5 is transformed into -> [0. 0. 0. 0. 0. 1. 0.]
validation_labels_one_hot = tf.keras.utils.to_categorical(validation_labels)

training_images, validation_images, test_images = training_images / 255.0, validation_images / 255.0, test_images / 255.0   #for a better and faster operation
                                                                                                            # we divide the value of the pixel to the max value that a pixel can get

# model = models.Sequential()
# model.add(layers.Conv2D(32, 2, padding="same",activation="relu", input_shape=(16, 16, 3)))
# model.add(layers.MaxPooling2D())
# 
# model.add(layers.Conv2D(32, 2,padding="same", activation="relu"))
# model.add(layers.MaxPooling2D())
#
# model.add(layers.Conv2D(64, 2,padding="same", activation="relu"))
# model.add(layers.MaxPooling2D())
# model.add(layers.Dropout(0.6))
#
# model.add(layers.Flatten())
# model.add(layers.Dense(128, activation="relu"))
# model.add(layers.Dense(10, activation="softmax"))
#
# model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
#
# hist = model.fit(training_images, training_labels,epochs=10, validation_data=(validation_images, validation_labels))
#
# loss, accuracy = model.evaluate(validation_images, validation_labels)
#
# print(f"Loss:{loss}")
# print(f"Accuracy:{accuracy}")

#PART2

model = models.Sequential()

model.add(layers.Conv2D(32, 2, activation="relu", input_shape=(16, 16, 3))) #here i played with the values to get a better accuracy and this is the best i found
model.add(layers.MaxPooling2D())

model.add(layers.Conv2D(32, 2, activation="relu"))
model.add(layers.MaxPooling2D())

model.add(layers.Flatten())
model.add(layers.Dense(256, activation="relu"))
model.add(layers.Dropout(0.25))

model.add(layers.Dense(128, activation="relu"))
model.add(layers.Dropout(0.25))

model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(7, activation="softmax"))

model.compile(loss= "categorical_crossentropy", optimizer="adam", metrics=["accuracy"]) #Compile defines the loss function, the optimizer and the metrics

hist = model.fit(training_images, training_labels_one_hot, epochs=15, batch_size=32,validation_data=(validation_images, validation_labels_one_hot)) #fit the tranin_data, train_labels
                                                                                                                                                     #and validate with validation_images and validation labels
                                                                                                                                                     #batch_size is to group images and to approximates the loss function and propagates the gradients back to update the weights

plt.plot(hist.history['accuracy'], label='accuracy')            #plotting accuracy
plt.plot(hist.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

plt.plot(hist.history['loss'], label='loss')                    #plotting loss
plt.plot(hist.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.ylim([1, 2])
plt.legend(loc='lower right')
plt.show()


loss, accuracy = model.evaluate(validation_images, validation_labels_one_hot)   #with this function we get the accuracy and loss

print(f"Loss:{loss}")
print(f"Accuracy:{accuracy}")
#

# FINISHED PART2

# model.save("image_classifier1.model")                 #pentru a nu mai rula proramul de fiecare data, il salvez si apoi import datele

# model = models.load_model("image_classifier.model")

pred1 = model.predict(test_images)                      #pred1 contains all predictions for test_images
predictions_test = []

for el in pred1:
    index = np.argmax(el)                               #using argmax we get the maximum index and we're using that index to get the actual class of the image
    predictions_test.append(class_names[index])

pred2 = model.predict(validation_images)                #pred 2 contains all predictions for validation images
predictions_val = []

for el in pred2:
    index = np.argmax(el)
    predictions_val.append(class_names[index])          #same here, i did this for the confusion matrix and for the accuracy and loss plot

def sample_submission(file_r, file_w):
    test_data = []
    with open(file_r) as r:                             #simply read all lines from file_r
        lines = r.readlines()
        for line in lines[1:]:                          #for each line (exept line[0] beacause it contains id, label string) we store in test_data the names of each image
            test_data.append(line.rstrip("\n"))         #line.rstrip is tu cut \n

    with open(file_w, "w") as w:                        #this is a file to output our submission
        w.write("id,label\n")                           #first line written in our submission
        for i in range(len(test_data)):                 #for each image, we write the name of the image and the class_name of this image
            w.write(f"{test_data[i]},{class_names[predictions_test[i]]}\n")


sample_submission("test.txt", "date_out.txt")            #call submission function

cf_matrix = confusion_matrix(validation_labels, predictions_val, labels=class_names)    #here we display the confusion matrix
f = seaborn.heatmap(cf_matrix, annot=True, fmt="d")
plt.show()


#MODEL 2

# training_images = np.array(training_images).reshape(len(training_images), -1) # convertion from 4D to 2D the svm model works with only 2D data
# training_labels = np.array(training_labels)
# validation_images = np.array(validation_images).reshape(len(validation_images), -1)
# validation_labels = np.array(validation_labels)
# test_images = np.array(test_images).reshape(len(test_images), -1)
#
# def normalize_data(train_data, test_data, type=None): #function to normalize data using sklearn library
#   if type == 'standard':
#     std_scaler = StandardScaler()
#     std_scaler.fit(train_data)
#     train_data = std_scaler.transform(train_data)
#     test_data = std_scaler.transform(test_data)
#   elif type =='l2':
#     normalized = Normalizer(norm='l2')
#     train_data = normalized.transform(train_data)
#     test_data = normalized.transform(test_data)
#   elif type =='l1':
#     normalized = Normalizer(norm='l1')
#     train_data = normalized.transform(train_data)
#     test_data = normalized.transform(test_data)
#
#   return train_data, test_data
#
# training_images, test_images = normalize_data(training_images, test_images)
#
# svm_model = svm.SVC(C=1,kernel= "linear")                 #create the actual model
# hist = svm_model.fit(training_images, training_labels)
#
# pred_validation_labels = svm_model.predict(validation_images) #get the predictions, this is to get the accuracy
# pred_test_labels = svm_model.predict(test_images) #this is the actual predictions that we need
#
# def sample_submision(file_r, file_w):                 #this function works same as the other one
#     test_data = []
#     with open(file_r) as r:
#         lines = r.readlines()
#         for line in lines[1:]:
#             test_data.append(line.rstrip("\n"))
#
#     with open(file_w, "w") as w:
#         w.write("id,label\n")
#         for i in range(len(test_data)):
#             w.write(f"{test_data[i]},{pred_test_labels[i]}\n")
#
# sample_submision("test.txt", "date_out.txt")
#
# cf_matrix = confusion_matrix(validation_labels, pred_validation_labels, labels=class_names)         #to display the confusion matrix
# f = seaborn.heatmap(cf_matrix, annot=True, fmt="d")
# plt.show()
#
# print("Accuracy:", accuracy_score(validation_labels, pred_validation_labels))                 #print the accuracy
# print("F1:", f1_score(validation_labels, pred_validation_labels, average=None))

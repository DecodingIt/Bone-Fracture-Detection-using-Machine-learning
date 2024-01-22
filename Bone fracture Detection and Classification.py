


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
import cv2
import imghdr
import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)




tf.config.list_physical_devices('GPU')


data_dir = 'archive' 



image_exts = ['jpeg','jpg', 'bmp', 'png']


data = tf.keras.utils.image_dataset_from_directory('archive', shuffle=False,)
data = data.shuffle(5000, seed=312, reshuffle_each_iteration=False)


data_iterator = data.as_numpy_iterator()


batch = data_iterator.next()



fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])



data = data.map(lambda x,y: (x/255, y))



data.as_numpy_iterator().next()[0].max()


data.as_numpy_iterator().next()[0].min()



len(data)




train_size = int(len(data)*.7)
val_size = int(len(data)*.2)
test_size = int(len(data)*.1)

train_size


train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)

len(test)



train





model = Sequential()




model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))



model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])



model.summary()




logdir='logs'



tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)



hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])



fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show()



fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
plt.show()


pre = Precision()
re = Recall()
acc = BinaryAccuracy()




for batch in test.as_numpy_iterator(): 
    X, y = batch
    yhat = model.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)




print(pre.result(), re.result(), acc.result())



image = cv2.imread('bro1.jpg')
plt.imshow(image)
plt.show()




resize = tf.image.resize(image, (256,256))
plt.imshow(resize.numpy().astype(int))
plt.show()




yhat = model.predict(np.expand_dims(resize/255, 0))


yhat



if yhat > 0.5: 
    print(f'Predicted class is a Healthy hand')
else:
    print(f'Predicted class is a Fractured Bone')






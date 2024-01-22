#!/usr/bin/env python
# coding: utf-8

# In[16]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
import cv2
import imghdr
import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy


# In[17]:


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)


# In[18]:


tf.config.list_physical_devices('GPU')


# In[48]:


data_dir = 'archive' 


# In[49]:


image_exts = ['jpeg','jpg', 'bmp', 'png']


# In[74]:


data = tf.keras.utils.image_dataset_from_directory('archive', shuffle=False,)
data = data.shuffle(5000, seed=312, reshuffle_each_iteration=False)


# In[75]:


data_iterator = data.as_numpy_iterator()


# In[76]:


batch = data_iterator.next()


# In[77]:


fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])


# In[54]:


data = data.map(lambda x,y: (x/255, y))


# In[55]:


data.as_numpy_iterator().next()[0].max()


# In[56]:


data.as_numpy_iterator().next()[0].min()


# In[57]:


len(data)


# In[58]:


train_size = int(len(data)*.7)
val_size = int(len(data)*.2)
test_size = int(len(data)*.1)


# In[59]:


train_size


# In[60]:


train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)


# In[61]:


len(test)


# In[62]:


train


# In[63]:


model = Sequential()


# In[64]:


model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# In[65]:


model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])


# In[66]:


model.summary()


# In[67]:


logdir='logs'


# In[68]:


tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)


# In[69]:


hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])


# In[70]:


fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show()


# In[49]:


fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
plt.show()


# In[50]:


pre = Precision()
re = Recall()
acc = BinaryAccuracy()


# In[51]:


for batch in test.as_numpy_iterator(): 
    X, y = batch
    yhat = model.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)


# In[52]:


print(pre.result(), re.result(), acc.result())


# In[12]:


image = cv2.imread('bro1.jpg')
plt.imshow(image)
plt.show()


# In[13]:


resize = tf.image.resize(image, (256,256))
plt.imshow(resize.numpy().astype(int))
plt.show()


# In[15]:


yhat = model.predict(np.expand_dims(resize/255, 0))


# In[61]:


yhat


# In[62]:


if yhat > 0.5: 
    print(f'Predicted class is a Healthy hand')
else:
    print(f'Predicted class is a Fractured Bone')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





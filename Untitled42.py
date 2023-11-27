#!/usr/bin/env python
# coding: utf-8

# In[18]:


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import BatchNormalization,Dense,Conv2D,MaxPooling2D,Activation,Flatten,Dropout


# In[19]:


train_path="C:/Users/Admin/Downloads/train-20231127T153200Z-001"
test_path="C:/Users/Admin/Downloads/test-20231127T152742Z-001"


# In[20]:


height, width = 150, 150
input_shape = (height, width, 3)


# In[21]:


train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)


# In[22]:


test_datagen = ImageDataGenerator(rescale=1./255)


# In[23]:


train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(height, width),
    batch_size=32,
    class_mode='binary'
)


# In[24]:


test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(height, width),
    batch_size=32,
    class_mode='binary'
)


# In[25]:


model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# In[26]:


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[27]:


history = model.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator
)


# In[28]:


test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')


# In[29]:


from tensorflow.keras.preprocessing import image
import numpy as np


# In[36]:


new_image_path = "C:/Users/Admin/Downloads/train-20231127T153200Z-001/train/cat.10.jpg"
new_image = image.load_img(new_image_path, target_size=(150,150))
new_image_array = image.img_to_array(new_image)
new_image_array = np.expand_dims(new_image_array, axis=0)
new_image_array /= 255.0


# In[37]:


prediction = model.predict(new_image_array)


# In[38]:


class_label = "Dog" if prediction > 0.5 else "Cat"

print(f"The model predicts that the image contains a {class_label} with confidence: {prediction[0][0]:.2f}")


# In[ ]:





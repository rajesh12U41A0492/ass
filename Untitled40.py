#!/usr/bin/env python
# coding: utf-8

# In[50]:


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import BatchNormalization,Dense,Conv2D,MaxPooling2D,Activation,Flatten,Dropout


# In[96]:


firess= os.listdir('C:/Users/Admin/Downloads/Fire-20231124T115135Z-001')
print (firess)
fires = []

for item in firess:
 all_fires= os.listdir('C:/Users/Admin/Downloads/Fire-20231124T115135Z-001'  + '/' +item)
 for fire in all_fires:
   fires.append((item, str('C:/Users/Admin/Downloads/Fire-20231124T115135Z-001' + '/' +item) + '/' + fire))
 print(fires)


# In[31]:


train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)


# In[43]:


train_dataset=train_datagen.flow_from_directory("C:/Users/Admin/Downloads/Fire-20231124T115135Z-001",target_size=(150,150),batch_size=32,class_mode='binary')


# In[44]:


test_dataset=test_datagen.flow_from_directory("C:/Users/Admin/Downloads/Fire-20231124T115135Z-001",target_size=(150,150),batch_size=32,class_mode='binary')


# In[45]:


class_indices=['fire','no_fire']


# In[46]:


test_dataset,class_indices


# In[67]:


model =Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))


# In[68]:


model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# In[69]:


model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])


# In[70]:


model.summary()


# In[71]:


r=model.fit(train_dataset,epochs=5,validation_data=test_dataset)


# In[74]:


predictions=model.predict(test_dataset)
predictions=np.round(predictions)


# In[75]:


predictions


# In[76]:


print(len(predictions))


# In[78]:


plt.plot(r.history['accuracy'])
plt.plot(r.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


# In[120]:


from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
def predict_image(image_path, model, height, width):
    img = image.load_img(image_path, target_size=(height, width))
    y = image.img_to_array(img)
    x = np.expand_dims(y, axis=0)
    val=model.predict(x)
    return val


# In[121]:


image_path = "C:/Users/Admin/Downloads/Fire-20231124T115135Z-001/Fire/1.jpg"
prediction_result = predict_image(image_path,model,150,150)

print("Prediction result:", prediction_result)


# In[ ]:





# In[ ]:





# In[ ]:





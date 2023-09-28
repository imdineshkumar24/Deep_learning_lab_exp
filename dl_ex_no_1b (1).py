#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install opendatasets')


# In[2]:


import opendatasets as od 


# In[3]:


od.download ("https://www.kaggle.com/datasets/sshikamaru/fruit-recognition")


# In[4]:


train_dir="fruit-recognition/train/train"
test_dir="fruit-recognition/test"


# In[5]:


from keras.preprocessing.image import ImageDataGenerator 
train_datagen= ImageDataGenerator (rescale=1./255,validation_split=0.2) 
test_datagen=ImageDataGenerator (rescale=1./255) 
size=128


# In[6]:


train_generator = train_datagen.flow_from_directory( 
train_dir, 
target_size=(100,100), 
color_mode="rgb", 
batch_size=size, 
class_mode='categorical', 
subset='training',                                                       
shuffle=True, 
seed=42)


# In[7]:


validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(100,100),
    color_mode="rgb",
    batch_size=size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)


# In[8]:


import matplotlib.pyplot as plt
import tensorflow
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Flatten


# In[30]:


model = Sequential()
model.add(Flatten(input_shape=(100,100,3)))

model.add(Dense(2048,activation='sigmoid'))
model.add(Dense(512,activation='sigmoid'))
model.add(Dense(33,activation='softmax'))


# In[31]:


model.summary()


# In[32]:


model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])


# In[33]:


mf=model.fit_generator(
    generator=train_generator,
    steps_per_epoch=len(train_generator),
    epochs=3,
    validation_data=validation_generator,
    validation_steps=len(validation_generator)
)


# In[34]:


train_acc = mf.history['accuracy'][-1]
print(train_acc)


# In[35]:


test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(100, 100),
        batch_size=size,
        color_mode="rgb",
        shuffle = "false",
        class_mode='categorical')


# In[36]:


y_prob=model.predict(test_generator)
y_pred=y_prob.argmax(axis=1)


# In[37]:


plt.plot(mf.history['loss'])
plt.plot(mf.history['val_loss'])
plt.plot(mf.history['accuracy'])
plt.plot(mf.history['val_accuracy'])


# In[38]:


y_test=test_generator.classes


# In[39]:


import numpy as np
individual_test_image = test_generator[0][0][4]
individual_test_label = test_generator[0][1][4]


plt.imshow(individual_test_image, cmap="gray")
plt.show()

prediction = model.predict(np.expand_dims(individual_test_image, axis=0)).argmax(axis=1)
print("Predicted class:", prediction)
print("True class:", individual_test_label.argmax())


# In[ ]:





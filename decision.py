import  streamlit as st
import numpy as np
import cv2
import pickle
from keras.utils import load_img,img_to_array


with open('GBC.pkl', 'rb') as mo:
    loadedModel = pickle.load(mo)

st.title("GBC")
classes = ['ABD' ,'SAM']



st.title('image classification')
img=st.file_uploader('select image',type=['jpg','png','jpeg'])
if img is not None:
	img=load_img(img, target_size=(100,100), color_mode='grayscale')
	imgarr=img_to_array(img)
	imgflatten=imgarr.flatten()
	st.write('flatten shape is ', imgflatten.shape)
	res=loadedModel.predict([imgflatten])
	st.write(f'model predicted class is : {classes[res[0]]}'); st.image(img, caption='Uploaded Image', width=300)

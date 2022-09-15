#Implemented by Priyanshi Sharma


import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import Polygon, Point
from matplotlib import pyplot as plt
import glob
import cv2
import tensorflow as tf 
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import datasets,layers,models 
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import os
from google.colab.patches import cv2_imshow
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from itertools import islice, count
import math 
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf 
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import datasets,layers,models 
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os
from keras.utils import np_utils
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Reshape, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50
from sklearn.metrics import confusion_matrix, classification_report
import itertools

trnames = glob.glob('/content/dataset/train1/*.jpg')
vnames=glob.glob('/content/dataset/valid/*.jpg')
tnames=glob.glob('/content/dataset/test/*.jpg')

slice_size = 224
counter = 0
def img_tile_label(imgname):
  img=[]
  indx=[]
  label=[]
  for imname in imgname:
      im = Image.open(imname)
      imr = np.array(im, dtype=np.uint8)
      height = imr.shape[0]
      width = imr.shape[1]
      labname = imname.replace('.jpg', '.txt')
      labels = pd.read_csv(labname, sep=' ', names=['class', 'x1', 'y1', 'w', 'h'])

      labels[['x1', 'w']] = labels[['x1', 'w']] * width
      labels[['y1', 'h']] = labels[['y1', 'h']] * height
      
      boxes = []
      
      for row in labels.iterrows():
          x1 = row[1]['x1'] - row[1]['w']/2
          y1 = (height - row[1]['y1']) - row[1]['h']/2
          x2 = row[1]['x1'] + row[1]['w']/2
          y2 = (height - row[1]['y1']) + row[1]['h']/2

          boxes.append((int(row[1]['class']), Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])))

      for i in range((height // slice_size)):
          for j in range((width // slice_size)):
              x1 = j*slice_size
              y1 = height - (i*slice_size)
              x2 = ((j+1)*slice_size) - 1
              y2 = (height - (i+1)*slice_size) + 1
              id=[(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
              id1=[(x1, y1,x2, y2)]
              indx.append(id1)
              pol = Polygon(id)
              imsaved = False
              slice_labels = []
              for box in boxes:
                  if pol.intersects(box[1]):
                      inter = pol.intersection(box[1])        
                      
                      if not imsaved:
                          sliced = imr[i*slice_size:(i+1)*slice_size, j*slice_size:(j+1)*slice_size]

                          img.append(sliced)
                          imsaved = True                    
                      
                      new_box = inter.envelope 
                      centre = new_box.centroid
                      x, y = new_box.exterior.coords.xy
                      new_width = (max(x) - min(x)) / slice_size
                      new_height = (max(y) - min(y)) / slice_size

                      new_x = (centre.coords.xy[0][0] - x1) / slice_size
                      new_y = (y1 - centre.coords.xy[1][0]) / slice_size
                      
                      counter += 1

                      slice_labels.append([box[0], new_x, new_y, new_width, new_height])
                      idx=[new_x, new_y, new_width, new_height]

              if len(slice_labels) > 0:
                  slice_df = pd.DataFrame(slice_labels, columns=['class', 'x1', 'y1', 'w', 'h'])
                  label.append(slice_df['class'][0])
              
              elif not imsaved:
                  sliced = imr[i*slice_size:(i+1)*slice_size, j*slice_size:(j+1)*slice_size]
                  img.append(sliced)
                  label.append(2)
                  imsaved = True
  return img,label,indx

train_img,train_label,tr_indx=img_tile_label(trnames)
valid_img,valid_label,vindx=img_tile_label(vnames)
test_img,test_label,tindx=img_tile_label(tnames)

v_img=[]
v_label=[]
for i in range(len(valid_img)):
  im=valid_img[i]
  if (im.shape== (224,224)):
    img_cv2_rgb = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    v_img.append(img_cv2_rgb)
    v_label.append(valid_label[i])

tr_img=[]
tr_label=[]
for i in range(len(train_img)):
  im=train_img[i]
  if (im.shape== (224,224)):
    img_cv2_rgb = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    tr_img.append(img_cv2_rgb)
    tr_label.append(train_label[i])
  else:
    tr_img.append(im)
    tr_label.append(train_label[i])

t_img=[]
t_label=[]
for i in range(len(test_img)):
  im=test_img[i]
  if (im.shape== (224,224)):
    img_cv2_rgb = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    t_img.append(img_cv2_rgb)
    t_label.append(test_label[i])
  else:
    t_img.append(im)
    t_label.append(test_label[i])

def img_numpy(img_list1):
  img=[]
  data_list=[]
  for i in range(0,len(img_list1)):
    im=img_list1[i]
    im=np.array(im, dtype="float32")
    img.append(im)
  data_list = np.array(img, dtype="float32")
  return data_list

train_img=img_numpy(tr_img)
test_img=img_numpy(t_img)
valid_img=img_numpy(v_img)

def binarise_label(lbl):
  lb1 = LabelBinarizer()
  lb = to_categorical(lbl)
  lb = np.array(lb)
  return lb

train_label=binarise_label(tr_label)
valid_label=binarise_label(v_label)
test_label=binarise_label(t_label)

(x_train, y_train), (x_test, y_test) = (train_img,train_label),(test_img,test_label)
train_im, valid_im, train_lab, valid_lab=train_img,valid_img,train_label,valid_label

class_types = ['text','nontext','bg'] 

training_data = tf.data.Dataset.from_tensor_slices((train_im, train_lab))
validation_data = tf.data.Dataset.from_tensor_slices((valid_im, valid_lab))
test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))

autotune = tf.data.AUTOTUNE 

train_data_batches = training_data.shuffle(buffer_size=40000).batch(128).prefetch(buffer_size=autotune)
valid_data_batches = validation_data.shuffle(buffer_size=10000).batch(32).prefetch(buffer_size=autotune)
test_data_batches = test_data.shuffle(buffer_size=10000).batch(32).prefetch(buffer_size=autotune)

class generate_patch(layers.Layer):
  def __init__(self, patch_size):
    super(generate_patch, self).__init__()
    self.patch_size = patch_size
    
  def call(self, images):
    batch_size = tf.shape(images)[0]
    patches = tf.image.extract_patches(images=images, 
                                       sizes=[1, self.patch_size, self.patch_size, 1], 
                                       strides=[1, self.patch_size, self.patch_size, 1], rates=[1, 1, 1, 1], padding="VALID")
    patch_dims = patches.shape[-1]
    patches = tf.reshape(patches, [batch_size, -1, patch_dims])  
    return patches

class generate_patch(layers.Layer):
  def __init__(self, patch_size):
    super(generate_patch, self).__init__()
    self.patch_size = patch_size
    
  def call(self, images):
    batch_size = tf.shape(images)[0]
    patches = tf.image.extract_patches(images=images, 
                                       sizes=[1, self.patch_size, self.patch_size, 1], 
                                       strides=[1, self.patch_size, self.patch_size, 1], rates=[1, 1, 1, 1], padding="VALID")
    patch_dims = patches.shape[-1]
    patches = tf.reshape(patches, [batch_size, -1, patch_dims]) 
    return patches

train_iter_7im, train_iter_7label = next(islice(training_data, 28, None))
train_iter_7im = tf.expand_dims(train_iter_7im, 0)
train_iter_7label = train_iter_7label.numpy()

patch_size=4 
generate_patch_layer = generate_patch(patch_size=patch_size)
patches = generate_patch_layer(train_iter_7im)

def render_image_and_patches(image, patches):
    plt.figure(figsize=(5, 5))
    plt.imshow(tf.cast(image[0], tf.uint8))
    plt.xlabel(class_types [np.argmax(train_iter_7label)], fontsize=13)
    n = int(np.sqrt(patches.shape[1]))
    plt.figure(figsize=(6, 6))
    for i, patch in enumerate(patches[0]):
        ax = plt.subplot(n, n, i+1)
        patch_img = tf.reshape(patch, (patch_size, patch_size, 3))
        ax.imshow(patch_img.numpy().astype("uint8"))
        ax.axis('off') 
        
render_image_and_patches(train_iter_7im, patches)

class PatchEncode_Embed(layers.Layer):
  
  def __init__(self, num_patches, projection_dim):
    super(PatchEncode_Embed, self).__init__()
    self.num_patches = num_patches
    self.projection = layers.Dense(units=projection_dim)
    self.position_embedding = layers.Embedding(
        input_dim=num_patches, output_dim=projection_dim)
    
  def call(self, patch):
    positions = tf.range(start=0, limit=self.num_patches, delta=1)
    encoded = self.projection(patch) + self.position_embedding(positions)
    return encoded

patch_encoder = PatchEncode_Embed(3136, 3136)(patches)

class generate_patch_conv(layers.Layer):
  
  def __init__(self, patch_size):
    super(generate_patch_conv, self).__init__()
    self.patch_size = patch_size

  def call(self, images):
    batch_size = tf.shape(images)[0]
    patches = layers.Conv2D(self.patch_size*self.patch_size*3, self.patch_size, self.patch_size, padding='valid')(images)

    patch_dims = patches.shape[-1]
    patches = tf.reshape(patches, [batch_size, -1, patch_dims])
    return patches  

class generate_patch_conv_orgPaper(layers.Layer):
  
  def __init__(self, patch_size, hidden_size):
    super(generate_patch_conv_orgPaper, self).__init__()
    self.patch_size = patch_size
    self.hidden_size = hidden_size

  def call(self, images):
    patches = layers.Conv2D(self.hidden_size, self.patch_size, self.patch_size, padding='valid', name='Embedding')(images) 
    rows_axis, cols_axis = (1, 2) 
    seq_len = (images.shape[rows_axis] // patch_size) * (images.shape[cols_axis] // patch_size)
    x = tf.reshape(patches, [-1, seq_len, self.hidden_size])
    return x

def generate_patch_conv_orgPaper_f(patch_size, hidden_size, inputs):
  patches = layers.Conv2D(filters=hidden_size, kernel_size=patch_size, strides=patch_size, padding='valid')(inputs)
  row_axis, col_axis = (1, 2)
  seq_len = (inputs.shape[row_axis] // patch_size) * (inputs.shape[col_axis] // patch_size)
  x = tf.reshape(patches, [-1, seq_len, hidden_size])
  return x

train_iter_7im = tf.cast(train_iter_7im, dtype=tf.float16)
generate_patch_conv_layer = generate_patch_conv(patch_size=patch_size)
patches_conv = generate_patch_conv_layer(train_iter_7im)
    
generate_patch_conv_orgPaper_layer = generate_patch_conv_orgPaper(patch_size=patch_size, hidden_size=64)
patches_conv_org = generate_patch_conv_orgPaper_layer(train_iter_7im)

hidden_size=64
patches_conv_org_f = generate_patch_conv_orgPaper_f(patch_size, hidden_size, train_iter_7im)

def render_image_and_patches(image, patches):
    plt.figure(figsize=(6, 6))
    plt.imshow(tf.cast(image[0], tf.uint8))
    plt.xlabel(class_types [np.argmax(train_iter_7label)], fontsize=13)
    n = int(np.sqrt(patches.shape[1]))
    plt.figure(figsize=(6, 6))

    for i, patch in enumerate(patches[0]):
        ax = plt.subplot(n, n, i+1)
        patch_img = tf.reshape(patch, (patch_size, patch_size, 3))
        ax.imshow(patch_img.numpy().astype("uint8"))
        ax.axis('off')

render_image_and_patches(train_iter_7im, patches_conv)

class AddPositionEmbs(layers.Layer):

  def __init__(self, posemb_init=None, **kwargs):
    super().__init__(**kwargs)
    self.posemb_init = posemb_init

  def build(self, inputs_shape):
    pos_emb_shape = (1, inputs_shape[1], inputs_shape[2])
    self.pos_embedding = self.add_weight('pos_embedding', pos_emb_shape, initializer=self.posemb_init)

  def call(self, inputs, inputs_positions=None):
    pos_embedding = tf.cast(self.pos_embedding, inputs.dtype)

    return inputs + pos_embedding

def mlp_block_f(mlp_dim, inputs):
  x = layers.Dense(units=mlp_dim, activation=tf.nn.gelu)(inputs)
  x = layers.Dropout(rate=0.1)(x) 
  x = layers.Dense(units=inputs.shape[-1], activation=tf.nn.gelu)(x)
  x = layers.Dropout(rate=0.1)(x)
  return x
def Encoder1Dblock_f(num_heads, mlp_dim, inputs):
  x = layers.LayerNormalization(dtype=inputs.dtype)(inputs)
  x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=inputs.shape[-1], dropout=0.1)(x, x) 
  x = layers.Add()([x, inputs]) 
  
  y = layers.LayerNormalization(dtype=x.dtype)(x)
  y = mlp_block_f(mlp_dim, y)
  y_1 = layers.Add()([y, x])  
  return y_1

rescale_layer = tf.keras.Sequential([layers.experimental.preprocessing.Rescaling(1./255)])

data_augmentation = tf.keras.Sequential([
  layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
  layers.experimental.preprocessing.RandomRotation(0.2), 
  layers.experimental.preprocessing.RandomZoom(height_factor=(0.2, 0.3), width_factor=(0.2, 0.3)),
  layers.experimental.preprocessing.RandomTranslation(0.3, 0.3, fill_mode='reflect', interpolation='bilinear',)
])


train_ds = (training_data.shuffle(40000).batch(32).map(lambda x, y: (data_augmentation(x), y), num_parallel_calls=autotune).prefetch(autotune))
valid_ds = (validation_data.shuffle(10000).batch(16).prefetch(autotune))

def Encoder_f(num_layers, mlp_dim, num_heads, inputs):
  x = AddPositionEmbs(posemb_init=tf.keras.initializers.RandomNormal(stddev=0.02), name='posembed_input')(inputs)
  x = layers.Dropout(rate=0.2)(x)
  for _ in range(num_layers):
    x = Encoder1Dblock_f(num_heads, mlp_dim, x)

  encoded = layers.LayerNormalization(name='encoder_norm')(x)
  return encoded

transformer_layers = 12
patch_size = 32
hidden_size = 512
num_heads = 12
mlp_dim = 1024

def build_ViT():
  inputs = layers.Input(shape=train_im.shape[1:])
  rescale = rescale_layer(inputs)
  patches = generate_patch_conv_orgPaper_f(patch_size, hidden_size, rescale)
  encoder_out = Encoder_f(transformer_layers, mlp_dim, num_heads, patches)  
  im_representation = tf.reduce_mean(encoder_out, axis=1)  # (1,) or (1,2)

  logits = layers.Dense(units=len(class_types), name='head', kernel_initializer=tf.keras.initializers.zeros)(im_representation)  

  final_model = tf.keras.Model(inputs = inputs, outputs = logits)
  return final_model

ViT_model = build_ViT()

ViT_model.compile(optimizer=tf.keras.optimizers.experimental.AdamW(learning_rate=10e-3), 
                  loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True), 
                  metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy"), tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top5 acc')]) 

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8,
                              patience=5, lr=1e-5, verbose=1)

ViT_Train = ViT_model.fit(train_ds, 
                        epochs = 100, 
                        validation_data=valid_ds, callbacks=[reduce_lr])

loss = ViT_Train.history['loss']
v_loss = ViT_Train.history['val_loss']

acc = ViT_Train.history['accuracy'] 
v_acc = ViT_Train.history['val_accuracy']

top5_acc = ViT_Train.history['top5 acc']
val_top5_acc = ViT_Train.history['val_top5 acc']
epochs = range(len(loss))

fig = plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.yscale('log')
plt.plot(epochs, loss, linestyle='--', linewidth=3, color='orange', alpha=0.7, label='Train Loss')
plt.plot(epochs, v_loss, linestyle='-.', linewidth=2, color='lime', alpha=0.8, label='Valid Loss')
plt.xlabel('Epochs', fontsize=11)
plt.ylabel('Loss', fontsize=12)
plt.legend(fontsize=12)
plt.subplot(1, 3, 2)
plt.plot(epochs, acc, linestyle='--', linewidth=3, color='orange', alpha=0.7, label='Train Acc')
plt.plot(epochs, v_acc, linestyle='-.', linewidth=2, color='lime', alpha=0.8, label='Valid Acc') 
plt.xlabel('Epochs', fontsize=11)
plt.ylabel('Accuracy', fontsize=12)
plt.legend(fontsize=12)
plt.subplot(1, 3, 3)
plt.plot(epochs, top5_acc, linestyle='--', linewidth=3, color='orange', alpha=0.7, label='Train Top 5 Acc')
plt.plot(epochs, val_top5_acc, linestyle='-.', linewidth=2, color='lime', alpha=0.8, label='Valid Top5 Acc') 
plt.xlabel('Epochs', fontsize=11)
plt.ylabel('Top5 Accuracy', fontsize=12)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()

predId = ViT_model.predict(x_test, batch_size=32)
predId = np.argmax(predId, axis=1)
print(classification_report(y_test.argmax(axis=1), predId,))

data=[]
label=[]
data_path = '/content/text/'
categories = os.listdir(data_path)
for category in categories:
    path = os.path.join(data_path, category)
    for img in os.listdir(path):
      if '.jpg' in img:
        img_path = os.path.join(path, img)
        image = load_img(img_path, target_size=(224, 224))
        image = img_to_array(image)
        data.append(image)
        label.append(category)

data = np.array(data, dtype="float32")

for i in range(len(label)):
  if(label[i]=='handwritten'):
    label[i]=0
  else:label[i]=1

label=binarise_label(label)

(trainX, testX, trainY, testY) = train_test_split(data, label,
	test_size=0.20, stratify=label, random_state=42)
(trainX, validX, trainY, validY) = train_test_split(trainX, trainY,
	test_size=0.20, stratify=trainY, random_state=42)

train_img=img_numpy(trainX)
test_img=img_numpy(testX)
valid_img=img_numpy(validX)
(x_train, y_train), (x_test, y_test) = (trainX,trainY),(testX,testY)
train_im, valid_im, train_lab, valid_lab=trainX,validX,trainY,validY

img_rows = 224
img_cols = 224
input_shape = (img_rows,img_cols,3)
epochs = 100
batch_size = 32

def getResNet50Model(lastFourTrainable=False):
    resnet_model = ResNet50(weights='imagenet', input_shape=input_shape, include_top=True)
    for layer in resnet_model.layers[:]:
        layer.trainable = False
    output = resnet_model.get_layer('avg_pool').output
    output = Flatten(name='new_flatten')(output)
    output = Dense(units=256, activation='relu', name='new_fc')(output)
    predictions = Dense(units=2, activation='softmax')(output)
    resnet_model = Model(resnet_model.input, predictions)
    if lastFourTrainable == True:
        resnet_model.get_layer('conv5_block3_2_bn').trainable = True
        resnet_model.get_layer('conv5_block3_3_conv').trainable = True
        resnet_model.get_layer('conv5_block3_3_bn').trainable = True
        resnet_model.get_layer('new_fc').trainable = True
    resnet_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return resnet_model

def plot_hist(history,title):
    plt.subplot(121)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(str(title)+' accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.subplot(122)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(str(title)+' loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

resnet_model_a = getResNet50Model(lastFourTrainable=False)

def score_train(model,test_x,test_y):
    train_scores = model.evaluate(test_x, test_y, verbose=1)

train_x_final_resnet=tensorflow.keras.applications.resnet.preprocess_input(trainX)
validation_x_final_resnet=tensorflow.keras.applications.resnet.preprocess_input(validX)
test_x_final_resnet=tensorflow.keras.applications.resnet.preprocess_input(testX)

history1 = resnet_model_a.fit(trainX, trainY, epochs=50,validation_data=(validX, validY))

score_train(resnet_model_a,testX,testY)

tname=glob.glob('/content/dataset/test2/*.jpg')

test_im,test_lbl,tidx=img_tile_label(tname)

xx_test=[]
yy_test=[]
for i in range(len(indices_pred)):
  if(abs(indices_pred[i][0][0]-indices_pred[i][0][2])==abs(tidx[i][0][0]-tidx[i][0][2])):
    xx_test.append(test_im[i])
    
    yy_test.append(test_lbl[i])

xx_test = np.array(xx_test, dtype="float32")
yy_test=binarise_label(yy_test)
score_train(resnet_model_a,xx_test,yy_test)

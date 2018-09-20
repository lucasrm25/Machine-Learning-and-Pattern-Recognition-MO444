import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

#import tensorflow as tf
#config = tf.ConfigProto(device_count = {'GPU': 0})
#sess = tf.Session(config=config)

from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import pickle

from keras import models
from keras import layers
from keras import optimizers
from keras.applications import VGG16, InceptionV3
from keras.preprocessing import image
from keras import backend as K #K.tensorflow_backend._LOCAL_DEVICES
from keras.callbacks import EarlyStopping


def image_to_dir(main_dir = './MO444_dogs/val'):
    for file in os.listdir(main_dir):
        if os.path.isfile(os.path.join(main_dir,file)):
            n_class = file[0:2]
            if not os.path.isdir(os.path.join(main_dir,n_class)):
                os.mkdir(os.path.join(main_dir,n_class))
            os.rename(os.path.join(main_dir,file), os.path.join(main_dir,n_class,file))
            print(file)


def org_test_data(main_dir = './MO444_dogs/test'):
    with open(os.path.join('./MO444_dogs/MO444_dogs_test.txt')) as f:
        content = f.readlines()
        for line in content:
            strs = line.split()
            if not os.path.isdir(os.path.join(main_dir,strs[-1])):
                os.mkdir(os.path.join(main_dir,strs[-1]))
            oldname = os.path.join( os.getcwd(),strs[-2] )
            newname = os.path.join( os.getcwd(), strs[-2].split('/')[0], strs[-2].split('/')[1], strs[-1] ,strs[-2].split('/')[2] )
            if not os.path.isfile(oldname):
                print('\nNAO EXISTE ',oldname)
            else:
                os.rename(oldname, newname)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          print_values=False,
                          print_ticks=False):
    import itertools
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    if print_ticks:
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

    if print_values:
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()



def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def create_CNN_InceptionV3(train_dir, validation_dir, image_size=299):
    inceptionV3_conv = InceptionV3(weights='imagenet', include_top=False, pooling=None, input_shape=(image_size, image_size, 3))   
    for layer in inceptionV3_conv.layers:
        layer.trainable = False
        print(layer, layer.trainable)
  
    model = models.Sequential()
    model.add(inceptionV3_conv)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(500,activation='sigmoid'))
    model.add(layers.Dense(83,activation='softmax'))
     
    model.summary()
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', #optimizers.RMSprop(lr=1e-4),
                  metrics=['acc', f1])   
    return model

def create_CNN_VGG(train_dir, validation_dir, image_size=224):    
    vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
    for layer in vgg_conv.layers:
        layer.trainable = False
        print(layer, layer.trainable)

    model = models.Sequential()
    model.add(vgg_conv)
    model.add(layers.Flatten())
    model.add(layers.Dense(500, activation='sigmoid'))
    #model.add(layers.Dropout(0.5))
#    model.add(layers.Dense(500, activation='sigmoid'))
#    model.add(layers.Dense(500, activation='sigmoid'))
    model.add(layers.Dense(83, activation='softmax'))
     
    model.summary()
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', #optimizers.RMSprop(lr=1e-4),
                  metrics=['acc', f1])   
    return model


def create_data_generator(train_dir, validation_dir, test_dir, image_size=299):
    # Data Augmentation    
    train_datagen = image.ImageDataGenerator(
          rescale=1/255,
          rotation_range=20,
          width_shift_range=0.2,
          height_shift_range=0.2,
          horizontal_flip=True,
          fill_mode='nearest')
     
    validation_datagen = image.ImageDataGenerator(rescale=1./255)
    
    test_datagen = image.ImageDataGenerator(rescale=1./255)
     
    # Change the batchsize according to your system RAM
    train_batchsize = 32
    val_batchsize = 10
     
    train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(image_size, image_size),
            batch_size=train_batchsize,
            class_mode='categorical')
     
    validation_generator = validation_datagen.flow_from_directory(
            validation_dir,
            target_size=(image_size, image_size),
            batch_size=val_batchsize,
            class_mode='categorical',
            shuffle=False)
    
    test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=(image_size, image_size),
            batch_size=val_batchsize,
            class_mode='categorical',
            shuffle=False)
    
    return train_generator, validation_generator, test_generator


#%% USER INPUT


train_dir = './MO444_dogs/train'
validation_dir = './MO444_dogs/val'
test_dir = './MO444_dogs/test'

experiment_name = 'InceptionV3_exp3_2denselayers'
load_model = False
train = True


if not os.path.isdir(os.path.join(os.getcwd(),experiment_name)):
    os.mkdir(os.path.join(os.getcwd(),experiment_name))

train_generator, validation_generator, test_generator = create_data_generator(train_dir, validation_dir, test_dir)

history=[]
if not load_model:
#    model = create_CNN_VGG(train_dir,validation_dir)
    model = create_CNN_InceptionV3(train_dir,validation_dir)
else:
    with open(os.path.join(os.getcwd(),experiment_name,'Training_History.pkl'), 'rb') as input:
        history = pickle.load(input)
    model = models.load_model(os.path.join(os.getcwd(),experiment_name,'CNN_dogs.h5'), custom_objects={'f1': f1})

if train:    
    early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')   
    for i in range(1):     
        history.append(model.fit_generator(
              train_generator,
              steps_per_epoch=train_generator.samples/train_generator.batch_size ,
              epochs=10,
              validation_data=validation_generator,
              validation_steps=validation_generator.samples/validation_generator.batch_size,
              verbose=1,
              use_multiprocessing=True,
              workers=0,
              callbacks = [early]).history)
        
    with open(os.path.join(os.getcwd(),experiment_name,'Training_History.pkl'), 'wb') as f:
        pickle.dump(history, f, pickle.HIGHEST_PROTOCOL)
    model.save(os.path.join(os.getcwd(),experiment_name,'CNN_dogs.h5'))


y_pred = np.argmax(model.predict_generator(validation_generator), axis=1)
y_true = validation_generator.classes

cnf_matrix = confusion_matrix(y_true, y_pred)
plot_confusion_matrix(cnf_matrix, classes=os.listdir(train_dir), normalize=True,
                      title='Normalized confusion matrix')


y_pred = np.argmax(model.predict_generator(test_generator), axis=1)
y_true = test_generator.classes

cnf_matrix = confusion_matrix(y_true, y_pred)
plot_confusion_matrix(cnf_matrix, classes=os.listdir(test_dir), normalize=True,
                      title='Normalized confusion matrix')

acc = np.sum(y_true == y_pred) / (len(y_pred))
from sklearn.metrics import f1_score
f1= f1_score(y_true, y_pred, average='macro')



if False:
    tr_cls = train_generator.classes
    val_cls = validation_generator.classes
    test_cls = test_generator.classes
    
    N = np.max(tr_cls)
    ind = np.arange(N)
    tr_dist = [np.sum(tr_cls==i) for i in ind]
    val_dist = [np.sum(val_cls==i) for i in ind]
    test_dist = [np.sum(test_cls==i) for i in ind]
    width = 0.8 
    
    p1 = plt.bar(ind, tr_dist, width)
    p2 = plt.bar(ind, val_dist, width, bottom=tr_dist)
    p3 = plt.bar(ind, test_dist, width, bottom=np.add(val_dist,tr_dist) )
    plt.xlabel('Class')
    plt.ylabel('Number of samples')
    plt.legend((p1[0], p2[0], p3[0]), ('Tr', 'Val', 'Test'), loc='upper right')





if False:
    fig = plt.figure(1)
    plt.plot(history[0]['val_acc'] + history[1]['val_acc'], label='Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.tight_layout()

#plt.imshow(train_generator[0][0][0])
#
#
#import matplotlib.pyplot as plt
#import cv2
#import numpy as np
#im = cv2.imread(os.path.join(train_dir,'00','00_0004.jpg'))
#im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
#plt.imshow(im)
#
#img_gen = train_datagen.flow(
#        [im],[1],
#        batch_size=train_batchsize)



if False:   # Testar foto individual
    import matplotlib.pyplot as plt
    import numpy as np
    
    img_path = os.path.join(train_dir,'00','00_0004.jpg')
    img_path = 'C:/Users/Renata/Desktop/fotoGabi.jpg'
    img_path = 'C:/Users/Renata/Desktop/fotoSusanne.jpg'
    img_path = 'C:/Users/Renata/Desktop/fotoHolly.jpg'
    
    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    pred = model.predict(x)
    print('Predicted: class', np.argmax(pred), 'with', np.max(pred), 'accuracy')


#acc = history.history['acc']
#val_acc = history.history['val_acc']
#loss = history.history['loss']
#val_loss = history.history['val_loss']
# 
#epochs = range(len(acc))
# 
#plt.plot(epochs, acc, 'b', label='Training acc')
#plt.plot(epochs, val_acc, 'r', label='Validation acc')
#plt.title('Training and validation accuracy')
#plt.legend()
# 
#plt.figure()
# 
#plt.plot(epochs, loss, 'b', label='Training loss')
#plt.plot(epochs, val_loss, 'r', label='Validation loss')
#plt.title('Training and validation loss')
#plt.legend()
# 
#plt.show()


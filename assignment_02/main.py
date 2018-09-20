import os
#import gzip
#import six.moves.cPickle as pickle
from keras.models import Sequential, model_from_json,model_from_yaml
from keras.layers import Dense, Activation
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD, Adagrad, Adadelta, Adamax

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler

import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from pyimageprocessing.localbinarypatterns import LocalBinaryPatterns
import pywt
from skimage.feature import greycomatrix, greycoprops
from skimage.measure import shannon_entropy
from scipy import ndimage, stats


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
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
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def image_statistics(Zin):
    #Input: Z, a 2D array, hopefully containing some sort of peak
    #Output: cx,cy,sx,sy,skx,sky,kx,ky
    #cx and cy are the coordinates of the centroid
    #sx and sy are the stardard deviation in the x and y directions
    #skx and sky are the skewness in the x and y directions
    #kx and ky are the Kurtosis in the x and y directions
    #Note: this is not the excess kurtosis. For a normal distribution
    #you expect the kurtosis will be 3.0. Just subtract 3 to get the
    #excess kurtosis.
    
    Z = Zin + Zin.min()

    h,w = np.shape(Z)

    x = range(w)
    y = range(h)


    #calculate projections along the x and y axes
    yp = np.sum(Z,axis=1)
    xp = np.sum(Z,axis=0)

    #centroid
    cx = np.sum(x*xp)/np.sum(xp)
    cy = np.sum(y*yp)/np.sum(yp)

    #standard deviation
    x2 = (x-cx)**2
    y2 = (y-cy)**2

    sx = np.sqrt( np.sum(x2*xp)/np.sum(xp) )
    sy = np.sqrt( np.sum(y2*yp)/np.sum(yp) )

    #skewness
    x3 = (x-cx)**3
    y3 = (y-cy)**3

    skx = np.sum(xp*x3)/(np.sum(xp) * sx**3)
    sky = np.sum(yp*y3)/(np.sum(yp) * sy**3)

    #Kurtosis
    x4 = (x-cx)**4
    y4 = (y-cy)**4
    kx = np.sum(xp*x4)/(np.sum(xp) * sx**4)
    ky = np.sum(yp*y4)/(np.sum(yp) * sy**4)


    return cx,cy,sx,sy,skx,sky,kx,ky


def plot_hist(ax, lbp_im):
    n_bins = np.size(lbp_im) 
    plt.bar(x=range(0, int(n_bins)), height=lbp_im.ravel(), color='r')
    ax.set_xlabel('LBP uniform patterns', fontsize=10)
    ax.set_ylabel('Percentage', fontsize=10)
    ax.grid(True)
#    ax.xaxis.label.set_size(20)

def process_images(img_folder):
    X_train = []
    y_train = []
    for idx, dirs in enumerate(os.listdir(img_folder)):
        print(idx, dirs)
        actual_folder = os.path.join(img_folder,dirs)
        for name in os.listdir(actual_folder):
            print('-- ',name)
    
            im = cv2.imread(os.path.join(actual_folder,name))
#            im_denoised = cv2.fastNlMeansDenoisingColored(im,None,1,10,7,21)
#            im_noise = im - im_denoised
#            im_gray_noise = cv2.cvtColor(im_noise, cv2.COLOR_BGR2GRAY)
#            
#            fig, ax = plt.subplots (1,3, sharex = False)
#            for axi in ax: axi.axis('off')
#            ax[0].imshow(im)
#            ax[1].imshow(im_denoised)
#            ax[2].imshow(im_noise)
#            fig.tight_layout()
            
            
            level=4
            distances = [1] #[1, 2, 3]
            angles = [0] #[0, np.pi/4, np.pi/2, 3*np.pi/4]
            properties = ['energy', 'homogeneity', 'contrast', 'energy', 'correlation']
            win_rows, win_cols = 3, 3
            sigma0 = 1e5  # degree of noise supression 

            features_WAV = np.array([])            
            coeffs_RGB = coeffs_RGB_filtered = [pywt.wavedec2(im[:,:,i], 'db8', level=level) for i in range(0,3)]
            im_filtered = np.empty(im.shape,dtype='uint8')
            for i_rgb in range(0,3):
                for i_level in range(1,level+1):
#                    print(i_level)
                    im_denoised = []
                    for i_vhd in range(0,3):
                        im_actual = coeffs_RGB[i_rgb][i_level][i_vhd]
                        
                        # High-order wavelet features
                        features_WAV = np.append(features_WAV, list(image_statistics(im_actual)))
                        
                        # Wavelet coefficient co-occurrence statistics
#                        im_gray_noise = cv2.cvtColor(coeffs_RGB[i_rgb][i_level][i_vhd], cv2.COLOR_BGR2GRAY)
#                        glcm = greycomatrix(coeffs_RGB[i_rgb][i_level][i_vhd], distances=distances, angles=angles, symmetric=True, normed=True)
#                        features_WAV = np.append(features_WAV, [greycoprops(coeffs_RGB[i_rgb][i_level][i_vhd], prop).ravel() for prop in properties])
#                        features_WAV = np.append(features_WAV, shannon_entropy(im_actual))
                        
                        # SPN   https://stackoverflow.com/questions/16107671/variance-image-in-python-using-gdal-and-a-running-window-approach
                        win_mean = ndimage.uniform_filter(im_actual,(win_rows,win_cols))
                        win_sqr_mean = ndimage.uniform_filter(im_actual**2,(win_rows,win_cols))
                        win_var = win_sqr_mean - win_mean**2
                          
                        im_denoised.append( im_actual*win_var**2/(win_var**2+sigma0) )
                    
                    coeffs_RGB_filtered[i_rgb][i_level] = tuple(im_denoised)
                im_filtered[:,:,i_rgb] = pywt.waverec2(coeffs_RGB_filtered[i_rgb], 'db8')

            im_noise = im-im_filtered
            
#            fig, ax = plt.subplots (1,3, sharex = False)
#            for axi in ax: axi.axis('off')
#            ax[0].imshow(im)
#            ax[1].imshow(im_filtered)
#            ax[2].imshow(im_noise)
#            fig.tight_layout()
            
            
#            lbp = LocalBinaryPatterns(24, 8)
#            lbp_im = lbp.describe(cv2.cvtColor(im_noise, cv2.COLOR_BGR2GRAY))
#            fig, ax = plt.subplots (1, figsize=(6, 4), sharex = True)
#            plot_hist(ax,lbp_im)
#            plt.tight_layout()
            
            features_SPN = np.array([])
            features_LBP = np.array([])
            lbp = LocalBinaryPatterns(24, 8)
            coeffs_RGB = [pywt.wavedec2(im_noise[:,:,i], 'db8', level=1) for i in range(0,3)]
            for i_rgb in range(0,3):
                for i_vhd in range(0,3):
                    for i_mom in range(1,9):
                        features_SPN = np.append(features_SPN, stats.moment( coeffs_RGB[i_rgb][1][i_vhd].flatten(), moment=i_mom))
                    features_LBP = np.append(features_LBP, lbp.describe(coeffs_RGB[i_rgb][1][i_vhd]))
                      
            X_train.append(np.concatenate((features_WAV,features_SPN,features_LBP)).tolist())
            y_train.append(idx)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    # REMOVE INVALID FEATURES
    bool_invalid_Features = np.any(np.isfinite(X_train), axis=0)
    idx_invalid_Features = np.where(np.logical_not( bool_invalid_Features ))
    X_train = np.delete(X_train, idx_invalid_Features, axis=1)
    
    return (X_train, y_train)


def build_logistic_model(input_dim, output_dim):
    model = Sequential()
    model.add(Dense(output_dim, input_dim=input_dim, activation='sigmoid'))
    model.summary()
    # compile the model
#    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#    adamax = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
    return model

def build_neuralnetwork_model(input_dim, output_dim):
    model = Sequential()
    model.add(Dense(200, input_dim=input_dim, activation='sigmoid'))
    model.add(Dense(output_dim, activation='sigmoid'))
    model.summary()
    # compile the model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['binary_accuracy'])
    return model


def train_model(X_train, y_train):
    # Split the data into a training set and a test set
    #X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
      
    # NORMALIZE DATA
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_norm = scaler.fit_transform(X_train)
    #X_retransformed = scaler.inverse_transform(X_transformed)
     
    batch_size = X_train_norm.shape[0]
    nb_classes = y_train.max() +1
    nb_epoch = 300
    nb_trains = 100
    input_dim = X_train_norm.shape[1]
       
    print(X_train_norm.shape[0], 'train samples')
    
    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
       
    model = []
    history = []
    score = []
    earlyStopping = EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')
    for i_class in range(nb_classes):
        model.append(build_logistic_model(input_dim, 1))
    #    model.append(build_neuralnetwork_model(input_dim, 1))
        for i_train in range(nb_trains):
            model[i_class].fit(X_train_norm, Y_train[:,i_class],
                                batch_size=batch_size, epochs=nb_epoch,
                                verbose=1, validation_split=0.4,
                                callbacks=[earlyStopping])
        score.append(model[i_class].evaluate(X_train_norm, Y_train[:,i_class], verbose=0))
    return model, scaler



experiment_name = 'SO_WAVELET_LOGISTIC'
extract_features = False
train_new_model = True
img_folder = 'data/train'


save_folder = os.path.join(os.getcwd(), experiment_name)

if extract_features:
    (X_train, y_train) = process_images(img_folder)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    np.save(os.path.join(save_folder,'X_train') , X_train)
    np.save(os.path.join(save_folder,'y_train') , y_train)
else:
    X_train = np.load(os.path.join(save_folder,'X_train.npy'))
    y_train = np.load(os.path.join(save_folder,'y_train.npy'))


if train_new_model:
    model, scaler = train_model(X_train, y_train)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    for idx,model_i in enumerate(model):
        json_string = model_i.to_json()
        open(os.path.join(save_folder,'model_'+str(idx)+'.json'), 'w').write(json_string)
        model_i.save_weights(os.path.join(save_folder,'model_'+str(idx)+'.h5'))
        with open(os.path.join(save_folder,'scaler.pkl'), 'wb') as f:
            pickle.dump(scaler, f, pickle.HIGHEST_PROTOCOL)
else:
    model = []
    for idx in range(9):
        model.append(model_from_json(open(os.path.join(save_folder,'model_'+str(idx)+'.json')).read()))
        model[idx].load_weights(os.path.join(save_folder,'model_'+str(idx)+'.h5'))
    with open(os.path.join(save_folder,'scaler.pkl'), 'rb') as input:
        scaler = pickle.load(input)




X_train_norm = scaler.fit_transform(X_train)
y_pred = np.hstack( ( model[i_class].predict(X_train_norm) for i_class in range(9) ) )
y_pred = np.argmax(y_pred, axis=1)

accuracy = np.sum(y_train==y_pred)/len(y_train)

print('TRAINING ACCURACY: %0.1f%%' % (accuracy*100))
cnf_matrix = confusion_matrix(y_train, y_pred)
plot_confusion_matrix(cnf_matrix, classes=os.listdir(img_folder), normalize=True,
                      title='Normalized confusion matrix')

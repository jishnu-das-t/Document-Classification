import numpy as np
from PIL import Image
from glob import glob
from sklearn.datasets import load_files
from keras.utils import np_utils
from keras.preprocessing import image
from tqdm import tqdm
from jnet_model.jnet import JNet
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

def load_train_dataset(path):
    classes = [item[73:-1] for item in
              sorted(glob("/media/das/C31C-A701/Python_Training/Project_Plans/alexnet/dataset/train/*/"))]
    data = load_files(path)
    train_files = np.array(data['filenames'])
    train_targets = np_utils.to_categorical(np.array(data['target']), len(classes))
    return train_files, train_targets

def load_test_dataset(path):
    classes = [item[73:-1] for item in
               sorted(glob("/media/das/C31C-A701/Python_Training/Project_Plans/alexnet/dataset/test/*/"))]
    data = load_files(path)
    test_files = np.array(data['filenames'])
    test_targets = np_utils.to_categorical(np.array(data['target']), len(classes))
    return test_files, test_targets

def path_to_tensor(img_path):
    img = image.load_img(img_path, target_size=(500, 400))
    x = image.img_to_array(img)
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

train_files, train_targets = load_train_dataset('/media/das/C31C-A701/Python_Training/Project_Plans/alexnet/dataset/train')
valid_files, valid_targets = load_train_dataset('/media/das/C31C-A701/Python_Training/Project_Plans/alexnet/dataset/valid')
test_files,test_targets = load_test_dataset('/media/das/C31C-A701/Python_Training/Project_Plans/alexnet/dataset/test')

train_tensors = paths_to_tensor(train_files).astype('float32')/255
valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255

jnet_model = JNet.build_model(object)

checkpointer = ModelCheckpoint(filepath='/media/das/C31C-A701/Python_Training/Project_Plans/alexnet/weights_best_checkpoints/jnet/weights.best.from_scratch.hdf5',
                               verbose=1, save_best_only=True)

hist = jnet_model.fit(train_tensors, train_targets,
        validation_data=(valid_tensors, valid_targets),
        epochs=15, batch_size=10, callbacks=[checkpointer], verbose=1)
jnet_model.save('/media/das/C31C-A701/Python_Training/Project_Plans/alexnet/model_fullconfig/jnet/fullconfig.h5')

loss,acc = jnet_model.evaluate(test_tensors,test_targets,verbose=1)
print(acc *100)

history = hist
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
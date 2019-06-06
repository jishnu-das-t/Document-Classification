from keras.models import load_model
from glob import glob
import numpy as np
from keras.utils import np_utils
from sklearn.datasets import load_files
from keras.preprocessing import image
from tqdm import tqdm

def load_model_from_path(path):
    model = load_model(path)
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model
def load_test_dataset(path):
    classes = ["FaxCoverSheet","NonFaxCoverSheet"]
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

def predict(model,test_tensors,test_targets):
    # classes = model.evaluate(test_tensors,test_targets,verbose=1)
    classes = model.predict_proba(test_tensors,verbose=1)
    print(test_targets)
    print(classes)


test_files, test_targets = load_test_dataset('/media/das/C31C-A701/Python_Training/Project_Plans/alexnet/dataset/test')
test_tensors = paths_to_tensor(test_files).astype('float32')/255


model= load_model_from_path("/media/das/C31C-A701/Python_Training/Project_Plans/alexnet/model_fullconfig/jnet/91/fullconfig.h5")
predict(model,test_tensors,test_targets)



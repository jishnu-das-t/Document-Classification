from keras.models import Sequential
from keras.layers import Dense,Dropout, Flatten, Conv2D, MaxPooling2D

class JNet:

    def build_model(self):
        model = Sequential()
        model.add(Conv2D(filters=16, kernel_size=2, padding='same', activation='relu',
                         input_shape=(500, 400, 3)))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.2))
        model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.3))
        model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.4))
        model.add(Flatten())
        model.add(Dense(500, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(250, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(100, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(25, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='softmax'))
        model.summary()
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
# %%
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LSTM
from keras.utils import to_categorical
from keras import optimizers
# %%
train = pd.read_csv('train.csv')

# %%
train.head()
# %%

train = train.drop(columns=['PassengerId', 'Name',
               'Fare', 'Ticket', 'Cabin'])
train = train.dropna()
X = train.drop(columns = 'Survived')
y = train['Survived'].copy()

# %%
X
# %%
y.info()

# %%
le = LabelEncoder()

X['Sex'] = le.fit_transform(X['Sex'])
X['Embarked'] = le.fit_transform(X['Embarked'])
# %%
scaler = MinMaxScaler()
X['Age'] = scaler.fit_transform(X[['Age']])
X['Pclass'] = scaler.fit_transform(X[['Pclass']])
X['SibSp'] = scaler.fit_transform(X[['SibSp']])
X['Parch'] = scaler.fit_transform(X[['Parch']])
X['Embarked'] = scaler.fit_transform(X[['Embarked']])
# %%
# CNN
y = to_categorical(y)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# %%
X_train.shape, X_test.shape, y_train.shape, y_test.shape
# %%
model = Sequential(
    [
        Dense(9, input_shape=(6,), activation='relu'),
        Dense(15, activation='relu'),
        Dense(50, activation='relu'),
        Dense(2, activation="softmax"),
    ]
)


# %%

model.compile(loss='binary_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'])

history = model.fit(X_train, y_train,
                    epochs=50, batch_size=32,
                    validation_data=(X_test, y_test),
                    verbose=1, validation_split=0.4,
                    shuffle=True)
# %%
model.summary()
# %%


# Evaluate the model
score = model.evaluate(X_test, y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# %%
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
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


# %%

model_1 = Sequential(
    [
        Dense(9, input_shape=(6,), activation='relu'),
        Dropout(0.5),
        Dense(15, activation='relu'),
        Dropout(0.5),
        Dense(50, activation='relu'),
        Dense(2, activation="softmax"),
    ]
)

model_1.compile(loss='binary_crossentropy',
                optimizer='Adam',
                metrics=['accuracy'])

history_1 = model_1.fit(X_train, y_train,
                        epochs=50, batch_size=32,
                        validation_data=(X_test, y_test),
                        verbose=1, validation_split=0.4,
                        shuffle=True)

score_1 = model_1.evaluate(X_test, y_test)
print('Test loss:', score_1[0])
print('Test accuracy:', score_1[1])
plt.plot(history_1.history['accuracy'])
plt.plot(history_1.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history_1.history['loss'])
plt.plot(history_1.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# %%
# %%

model_2 = Sequential(
    [
        Dense(9, input_shape=(6,), activation='relu'),
        Dropout(0.5),
        Dense(15, activation='relu'),
        Dropout(0.5),
        Dense(50, activation='relu'),

        # Flatten(),
        Dense(2, activation="softmax"),
    ]
)

model_2.compile(loss='binary_crossentropy',
                optimizer='Adam',
                metrics=['accuracy'])

history_2 = model_2.fit(X_train, y_train,
                        epochs=50, batch_size=32,
                        validation_data=(X_test, y_test),
                        verbose=1, validation_split=0.4,
                        shuffle=True)

score_2 = model_2.evaluate(X_test, y_test)
print('Test loss:', score_1[0])
print('Test accuracy:', score_2[1])
plt.plot(history_2.history['accuracy'])
plt.plot(history_2.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history_2.history['loss'])
plt.plot(history_2.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# %%

model_3 = Sequential(
    [
        Dense(9, input_shape=(6,), activation='relu'),
        Dropout(0.5),
        Dense(15, activation='relu'),
        Dropout(0.5),
        Dense(50, activation='relu'),
        Dropout(0.5),
        Dense(2, activation="softmax"),
    ]
)

model_3.compile(loss='binary_crossentropy',
                optimizer='Adam',
                metrics=['accuracy'])

history_3 = model_3.fit(X_train, y_train,
                        epochs=50, batch_size=32,
                        validation_data=(X_test, y_test),
                        verbose=1, validation_split=0.4,
                        shuffle=True)

score_3 = model_3.evaluate(X_test, y_test)
print('Test loss:', score_1[0])
print('Test accuracy:', score_2[1])
plt.plot(history_3.history['accuracy'])
plt.plot(history_3.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history_3.history['loss'])
plt.plot(history_3.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# %%

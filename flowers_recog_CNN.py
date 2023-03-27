import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from pathlib import Path
from tensorflow.keras import models,layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, accuracy_score


folder_dir = Path('/Users/Asus/PycharmProjects/Flower_recog/dataset_flowers')

#儲存修改大小後的image
data = []
#儲存每一個image是存放在哪一個folder中(即花朵類別)
label = []
#image crop to 64X64
SIZE = 64

#讀取folder，並整理各image資料
for folder in os.listdir(folder_dir):
    for file in os.listdir(os.path.join(folder_dir, folder)):
        if file.endswith("jpg"):
            label.append(folder)
            img = cv2.imread(os.path.join(folder_dir, folder, file))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            im = cv2.resize(img_rgb, (SIZE,SIZE))
            data.append(im)
        else:
            continue


data_arr = np.array(data)
label_arr = np.array(label)


#將label分成五大類(由0,1表示)，並分別對圖片做標準化
encoder = LabelEncoder()
y = encoder.fit_transform(label_arr)
y = to_categorical(y,5)
X = data_arr/255

#split成test跟train database，test_size值控制要切割的標準
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.70, random_state=10)

#建立keras CNN模型
model = models.Sequential([
    #Each layer's filters數量由小到大，前面的卷積處理形狀、輪廓，後面為整塊或較複雜的理解。
    # First convolutional layer
    layers.Conv2D(16, kernel_size = (3,3),padding = 'Same',activation ='relu', input_shape = (SIZE,SIZE,3)),
    layers.MaxPooling2D(pool_size=(2,2)),
    # Second convolutional layer
    layers.Conv2D(32, kernel_size = (3,3),padding = 'Same',activation ='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    # Third convolutional layer
    layers.Conv2D(64, kernel_size = (3,3),padding = 'Same',activation ='relu'),
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.Dropout(rate=0.20),

    #將 feature轉為一維資料
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    #防止過度擬合
    layers.Dropout(rate=0.5),
    #output layer, 5 categorical
    layers.Dense(5, activation = "softmax")
])
model.summary()

#將image資料透過zoom, shift, flip, rotate增強，目的為增加CNN辨識率
datagen = ImageDataGenerator(
        rotation_range=20,
        zoom_range = 0.20,
        width_shift_range=0.3,
        height_shift_range=0.3,
        horizontal_flip=True,
        vertical_flip=True
)

#傳遞訓練數據集，計算實際執行轉換到image數據所需的任何統計參數
datagen.fit(X_train)

# Compiling the CNN，優化演算法選用adam，學習率0.0001，損失函數
model.compile(optimizer=Adam(lr=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])

batch_size=16
epochs=32

#對於增強的image進行批次訓練
model.fit_generator(datagen.flow(X_train,y_train, batch_size=batch_size),
                              epochs = epochs,
                              validation_data = (X_test,y_test),
                              verbose = 1)


#categories存放花朵的五個類別
categories = np.sort(os.listdir(folder_dir))

#製作subplot，預測結果視覺化
fig, ax= plt.subplots(2,5, figsize=(20, 35))
#actual -> 存放images的實際類別
actual=[]
#predicted -> 存放images的預測類別
predicted=[]

#製作2X5，也就是隨機10筆資料的預測
for i in range(2):
    for j in range(5):
        k = int(np.random.random_sample() * len(X_test))
        if(categories[np.argmax(y_test[k])] == categories[np.argmax(model.predict(X_test)[k])]):
            ax[i, j].set_title("TRUE: " + categories[np.argmax(y_test[k])] + "\nPRED: " + categories[
                np.argmax(model.predict(X_test)[k])], color='green', fontsize=6)
            ax[i,j].set_axis_off()
            ax[i,j].imshow(np.array(X_test)[k].reshape(SIZE, SIZE, 3), cmap='gray')
        else:
            ax[i, j].set_title("TRUE: " + categories[np.argmax(y_test[k])] + "\nPRED: " + categories[
                np.argmax(model.predict(X_test)[k])], color='red', fontsize=6)
            ax[i, j].set_axis_off()
            ax[i,j].imshow(np.array(X_test)[k].reshape(SIZE, SIZE, 3), cmap='gray')

        #將每個image實際類別與預測類別分別存放，以便之後製作confusion matrix
        actual.append(categories[np.argmax(y_test[k])])
        predicted.append(categories[np.argmax(model.predict(X_test)[k])])

scores = model.evaluate(X_test, y_test)
print(scores)
print("Test Accuracy = ",accuracy_score(actual,predicted))
plt.subplots_adjust(wspace=0.2,hspace=0.75)
plt.show()

#confusion matrix製作
confusion_mat = confusion_matrix(y_true=actual, y_pred=predicted,labels=categories)
fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confusion_mat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confusion_mat.shape[0]):
    for j in range(confusion_mat.shape[1]):
        ax.text(x=j, y=i, s=confusion_mat[i,j], va='center', ha='center')

plt.xlabel('predicted label', fontsize=7)
plt.ylabel('true label', fontsize=7)
plt.show()



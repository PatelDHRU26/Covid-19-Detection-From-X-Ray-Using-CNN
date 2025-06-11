from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
import os
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.BrowseImage = QtWidgets.QPushButton(self.centralwidget)
        self.BrowseImage.setGeometry(QtCore.QRect(160, 370, 151, 51))
        self.BrowseImage.setObjectName("BrowseImage")

        self.imageLbl = QtWidgets.QLabel(self.centralwidget)
        self.imageLbl.setGeometry(QtCore.QRect(200, 80, 361, 261))
        self.imageLbl.setFrameShape(QtWidgets.QFrame.Box)
        self.imageLbl.setText("")
        self.imageLbl.setObjectName("imageLbl")

        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(110, 20, 621, 20))
        font = QtGui.QFont()
        font.setFamily("Courier New")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")

        self.Classify = QtWidgets.QPushButton(self.centralwidget)
        self.Classify.setGeometry(QtCore.QRect(160, 450, 151, 51))
        self.Classify.setObjectName("Classify")

        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(430, 370, 111, 16))
        self.label.setObjectName("label")

        self.Training = QtWidgets.QPushButton(self.centralwidget)
        self.Training.setGeometry(QtCore.QRect(400, 450, 151, 51))
        self.Training.setObjectName("Training")

        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(400, 390, 211, 51))
        self.textEdit.setObjectName("textEdit")

        MainWindow.setCentralWidget(self.centralwidget)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.BrowseImage.clicked.connect(self.loadImage)
        self.Classify.clicked.connect(self.classifyFunction)
        self.Training.clicked.connect(self.trainingFunction)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "COVID-19 Detection"))
        self.BrowseImage.setText(_translate("MainWindow", "Browse Image"))
        self.label_2.setText(_translate("MainWindow", "            COVID-19 DETECTION"))
        self.Classify.setText(_translate("MainWindow", "Classify"))
        self.label.setText(_translate("MainWindow", "Recognized Class"))
        self.Training.setText(_translate("MainWindow", "Training"))

    def loadImage(self):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp);;All Files (*)") 
        if fileName:  
            self.file = fileName
            pixmap = QtGui.QPixmap(fileName)
            pixmap = pixmap.scaled(self.imageLbl.width(), self.imageLbl.height(), QtCore.Qt.KeepAspectRatio)
            self.imageLbl.setPixmap(pixmap)
            self.imageLbl.setAlignment(QtCore.Qt.AlignCenter)

    def classifyFunction(self):
        try:
            if not os.path.exists("model.json") or not os.path.exists("model.h5"):
                self.textEdit.setText("Model files not found! Train the model first.")
                return

            if not hasattr(self, 'file') or not self.file:
                self.textEdit.setText("Please load an image first.")
                return

            with open('model.json', 'r') as json_file:
                loaded_model_json = json_file.read()
            loaded_model = model_from_json(loaded_model_json)
            loaded_model.load_weights("model.h5")
            print("Loaded model from disk")

            label = ["Covid", "Normal"]
            test_image = load_img(self.file, target_size=(128, 128)) #pixel
            test_image = img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis=0)

            result = loaded_model.predict(test_image)
            label2 = label[result.argmax()]
            print("Predicted Label:", label2)

            self.textEdit.setText(label2)

        except Exception as e:
            print("Error:", e)
            self.textEdit.setText(f"Error: {str(e)}")

    def trainingFunction(self):
        try:
            self.textEdit.setText("Training under process...")
            model = Sequential([
                Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
                MaxPooling2D((2, 2)),
                BatchNormalization(),
                Conv2D(64, (3, 3), activation='relu'),
                MaxPooling2D((2, 2)),
                BatchNormalization(),
                Dropout(0.2),
                Flatten(),
                Dense(128, activation='relu'),
                Dropout(0.3),
                Dense(2, activation='softmax')
            ])

            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

            train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
            test_datagen = ImageDataGenerator(rescale=1./255)

            training_set = train_datagen.flow_from_directory('TrainingDataset', target_size=(128, 128), batch_size=8, class_mode='categorical')
            test_set = test_datagen.flow_from_directory('TestingDataset', target_size=(128, 128), batch_size=8, class_mode='categorical')

            model.fit(training_set, steps_per_epoch=100, epochs=10, validation_data=test_set, validation_steps=125)

            model_json = model.to_json()
            with open("model.json", "w") as json_file:
                json_file.write(model_json)
            model.save_weights("model.h5")
            print("Saved model to disk")
            self.textEdit.setText("Training complete! Model saved.")

        except Exception as e:
            print("Error:", e)
            self.textEdit.setText(f"Error: {str(e)}")

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

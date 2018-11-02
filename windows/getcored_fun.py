# -*- coding:utf-8 -*- 
# 

from PyQt5.QtWidgets import QWidget, QApplication, QMainWindow, QFileDialog, QMessageBox
from PyQt5.QtCore import QTimer, QSize

from getcored_ui import Ui_Form
import cv2 
import os
import copy
from identiffun.face_faster import GenerateClass


def img_reszie(img, size_w_h):# size_w_h = (width, height),
    return cv2.resize(img, size_w_h, interpolation=cv2.INTER_CUBIC)

def img_rotate(img, center, angle, size_w_h):
    M = cv2.getRotationMatrix2D(center, angle, 1)
    return cv2.warpAffine(img, M, size_w_h)

# flipCode – 翻转模式，
# flipCode==0垂直翻转（沿X轴翻转），
# flipCode>0水平翻转（沿Y轴翻转），
# flipCode<0水平垂直翻转（先沿X轴翻转，再沿Y轴翻转，等价于旋转180°）
def img_flip(img, flipCode):
    return cv2.flip(img , flipCode)


class getDataWindows(QWidget):
    def __del__(self):
        if hasattr(self, "camera"):
            self.camera.release()# 释放资源

    def __init__(self):
        super(getDataWindows, self).__init__()
        self.window = Ui_Form()
        self.window.setupUi(self)

        self.timer = QTimer()# 定义一个定时器对象
        self.timer.timeout.connect(self.timer_fun) #计时结束调用方法

        self.window.openUSBBtn.clicked.connect(self.timer_start)
        self.window.closeUSBBtn.clicked.connect(self.closeBtn_fun)

        self.window.fbl_comboBox.currentIndexChanged.connect(self.set_width_and_height)

        self.window.capBtn.clicked.connect(self.catch_picture)
        self.window.saveBtn.clicked.connect(self.saveBtn_fun)

        self.window.openBtn.clicked.connect(self.openBtn_fun)
        self.window.sfBtn.clicked.connect(self.sfBtn_fun)
        self.window.xzBtn.clicked.connect(self.xzBtn_fun)
        self.window.spBtn.clicked.connect(self.spBtn_fun)
        self.window.czBtn.clicked.connect(self.czBtn_fun)

        self.window.face_1_checkBox.clicked.connect(self.face_1_checkBox_fun)

        self.window.faceCheckBox.clicked.connect(self.faceCheckBox_fun)
        self.getface_flag = False

        self.mygener = GenerateClass(".")
        self.knn_clf = self.mygener.get_knn_clf("./identiffun/trained_knn_model1.clf")

        self.window.pushFaceBtn.clicked.connect(self.pushFaceBtn_fun)

    def pushFaceBtn_fun(self):
        try:
            pass
            # 1. 将相应的图片保存到对应的目录下
            # 2. 更新clf文件即可
        except Exception as e:
            QMessageBox.warning(self, "ERROR", str(e), QMessageBox.Cancel)

    def faceCheckBox_fun(self):
        if self.window.faceCheckBox.isChecked():
            self.getface_flag = True
        else:
            self.getface_flag = False

    def face_1_checkBox_fun(self):
        if self.window.face_1_checkBox.isChecked():
            if hasattr(self, "raw_frame"):
                predictions = self.mygener.predict(self.raw_frame, self.knn_clf)
                self.showimg2picfigaxes(self.mygener.show_prediction_labels_on_image(self.raw_frame, predictions))



    def czBtn_fun(self):
        if hasattr(self, "raw_frame"):
            self.reszie_img = img_flip(copy.deepcopy(self.raw_frame), 0)
            self.showimg2picfigaxes(self.reszie_img)

    def spBtn_fun(self):
        if hasattr(self, "raw_frame"):
            self.reszie_img = img_flip(copy.deepcopy(self.raw_frame), 1)
            self.showimg2picfigaxes(self.reszie_img)

    def xzBtn_fun(self):
        if hasattr(self, "raw_frame"):
            X = self.window.xz_X_spinBox.value()
            Y = self.window.xz_Y_spinBox.value()
            width = self.window.sf_W_spinBox.value()
            height = self.window.sf_H_spinBox.value()
            angle = self.window.xz_D_spinBox.value()
            self.reszie_img = img_rotate(copy.deepcopy(self.raw_frame), (X,Y), angle, (width, height))
            self.showimg2picfigaxes(self.reszie_img)

    def sfBtn_fun(self):
        if hasattr(self, "raw_frame"):
            width = self.window.sf_W_spinBox.value()
            height = self.window.sf_H_spinBox.value()
            self.reszie_img = img_reszie(copy.deepcopy(self.raw_frame), (width, height))
            self.showimg2picfigaxes(self.reszie_img)

    def init_window_info(self):
        self.window.sf_W_spinBox.setValue(self.raw_frame.shape[1])
        self.window.sf_H_spinBox.setValue(self.raw_frame.shape[0])
        self.window.xz_X_spinBox.setValue(self.raw_frame.shape[1]//2)
        self.window.xz_Y_spinBox.setValue(self.raw_frame.shape[0]//2)

    def openBtn_fun(self):
        filename, filetype = QFileDialog.getOpenFileName(self, "Open", None, "jpg files(*.jpg);;All Files(*)")
        if filename:
            self.raw_frame = cv2.imread(filename)   
            self.init_window_info()         
            self.showimg2picfigaxes(self.raw_frame)

    def catch_picture(self):
        if hasattr(self, "camera") and self.camera.isOpened():
            ret, frame = self.camera.read()
            if ret:
                self.raw_frame = copy.deepcopy(frame)
                self.init_window_info() 
                self.showimg2picfigaxes(frame)
            else:
                pass # get faild

    def saveBtn_fun(self):
        filename, filetype = QFileDialog.getSaveFileName(self, "save", "", "jpg Files(*.jpg)::All Files(*)")
        if filename:
            if hasattr(self, "raw_frame"):
                if hasattr(self, "reszie_img"):
                    cv2.imwrite(filename, self.reszie_img)
                else:
                    cv2.imwrite(filename, self.raw_frame)


    def set_width_and_height(self):
        width, height = self.window.fbl_comboBox.currentText().split('*')
        if hasattr(self, "camera"):
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))

    def closeBtn_fun(self):
        if hasattr(self, "camera"):
            self.camera.release()# 释放资源
            self.timer.stop()
        self.window.figaxes_video.clear()
        self.window.figure_video.canvas.draw()

    def timer_fun(self):
        ret, frame = self.camera.read()
        if ret:
            self.showimg2videofigaxes(frame)
        else:
            self.timer.stop()

    def timer_start(self):
        if hasattr(self, "camera"):
            if not self.camera.isOpened():
                self.camera.open(0)
        else:
            self.camera = cv2.VideoCapture(0)

        if self.camera.isOpened():
            pass
        else:
            print("not Open USB")
            return
        self.get_camera_params()
        self.timer.start(41) #设置计时间隔并启动

    def get_camera_params(self):
        width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.window.fbl_comboBox.setCurrentText("%d*%d" % (int(width), int(height)))

        fps = self.camera.get(cv2.CAP_PROP_FPS)
        if fps == float('inf'):
            self.window.zl_SpinBox.setValue(0.0)
        else:
            self.window.zl_SpinBox.setValue(fps)

        brightness = self.camera.get(cv2.CAP_PROP_BRIGHTNESS)
        if brightness == float('inf'):
            self.window.ld_SpinBox.setValue(0.0)
        else:
            self.window.ld_SpinBox.setValue(brightness)

        contrast = self.camera.get(cv2.CAP_PROP_CONTRAST)
        if contrast == float('inf'):
            self.window.dbd_SpinBox.setValue(0.0)
        else:
            self.window.dbd_SpinBox.setValue(contrast)

        hue = self.camera.get(cv2.CAP_PROP_HUE)
        if hue == float('inf'):
            self.window.sd_SpinBox.setValue(0.0)
        else:
            self.window.sd_SpinBox.setValue(hue)

        exposure =self.camera.get(cv2.CAP_PROP_EXPOSURE) 
        if exposure == float('inf'):
            self.window.bg_SpinBox.setValue(0.0)
        else:
            self.window.bg_SpinBox.setValue(exposure) # inf

        saturation =self.camera.get(cv2.CAP_PROP_SATURATION) 
        if saturation == float('inf'):
            self.window.bhd_SpinBox.setValue(0.0)
        else:
            self.window.bhd_SpinBox.setValue(saturation) # inf

    def showimg2videofigaxes(self, img):
        if self.getface_flag:
            predictions = self.mygener.predict(img, self.knn_clf)
            ret_img = self.mygener.show_prediction_labels_on_image(img, predictions)
        else:
            ret_img = img
        b, g, r = cv2.split(ret_img)
        imgret = cv2.merge([r,g,b])# 这个就是前面说书的，OpenCV和matplotlib显示不一样，需要转换
        self.window.figaxes_video.clear()
        self.window.figaxes_video.imshow(imgret)
        self.window.figure_video.canvas.draw()

    def showimg2picfigaxes(self,img):
        b, g, r = cv2.split(img)
        imgret = cv2.merge([r,g,b])# 这个就是前面说书的，OpenCV和matplotlib显示不一样，需要转换
        self.window.figaxes_pic.clear()
        self.window.figaxes_pic.imshow(imgret)
        self.window.figure_pic.canvas.draw()


if __name__ == '__main__':
    
    import sys
    app = QApplication(sys.argv)
    ui = getDataWindows()
    ui.show()
    sys.exit(app.exec_())
运行环境：
windows下安装的VMware虚拟机(lubuntu18.04(64bit) + python3.6)
安装软件： 
1. sudo apt-get install python3-opencv
2. sudo apt-get install python3-matplotlib
3. pip3 install opencv-contrib-python 
# 安装完opencv-contrib-python的时候，打不开摄像头，将虚拟机的USB设置成3.0的即可

4.1 sudo apt-get install cmake 
4.2 sudo apt-get install libboost-python-dev
	4.2.1 dlib需要源码安装速度会快一些
4.3 pip3 install face_recognition 
	4.3.1 sudo apt install python3-sklearn

pyQt5:
sudo apt-get install python3-pyqt5
sudo apt-get install pyqt5-dev-tools
sudo apt-get install qttools5-dev-tools
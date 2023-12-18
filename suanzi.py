import cv2 as cv
import numpy as np

# 加载图像
img = cv.imread('shuzi\dog.jpg', 0)
cv.imshow('jpg', img)
cv.waitKey(0)

# prewitt
kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
ky = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
imgX = cv.filter2D(img, cv.CV_64F, kx)
imgY = cv.filter2D(img, cv.CV_64F, ky)
imgXY = np.sqrt(imgX ** 2 + imgY ** 2)
imgXY2 = np.abs(imgX) + np.abs(imgY)

# Roberts
kx = np.array([[-1, 0], [0, 1]], dtype=np.float32)
ky = np.array([[0, -1], [1, 0]], dtype=np.float32)
imgX = cv.filter2D(img, cv.CV_64F, kx)
imgY = cv.filter2D(img, cv.CV_64F, ky)
imgXY = np.sqrt(imgX ** 2 + imgY ** 2)
imgXY2 = np.abs(imgX) + np.abs(imgY)

# 显示结果
cv.imshow('prewitt imgXY', imgXY)
cv.waitKey(0)
cv.imshow('Roberts imgXY2', imgXY2)
cv.waitKey(0)
cv.destroyAllWindows()

# 加载图像
img = cv.imread('shuzi\dog.jpg', 0)
cv.imshow('jpg', img)
cv.waitKey(0)

# Sobel 
imgX = cv.Sobel(img, cv.CV_16S, 1, 0)
imgY = cv.Sobel(img, cv.CV_16S, 0, 1)
imgXY = np.abs(imgX) + np.abs(imgY)

# 显示结果
cv.imshow('Sobel imgXY', imgXY)
cv.waitKey(0)
cv.destroyAllWindows()

# 加载图像
img = cv.imread('shuzi\dog.jpg', 0)
cv.imshow('jpg', img)
cv.waitKey(0)

# Laplacian 和 GaussianBlur 
img_lap = cv.Laplacian(img, cv.CV_64F)
cv.imshow('Laplacian ', np.abs(img_lap))
cv.waitKey(0)

img_blur = cv.GaussianBlur(img, (3, 3), 1)
img_log = cv.Laplacian(img_blur, cv.CV_64F)
cv.imshow('Log ', np.abs(img_log))
cv.waitKey(0)
cv.destroyAllWindows()



import cv2 as cv
import numpy as np

img = cv.imread('shuzi\dog.jpg', 0)
cv.imshow('jpg', img)
cv.waitKey(0)

# 1. 平滑
img_blur = cv.GaussianBlur(img, (5, 5), 2)

# 2. 计算梯度
gradx = cv.Sobel(img_blur, cv.CV_64F, 1, 0)
grady = cv.Sobel(img_blur, cv.CV_64F, 0, 1)
R = np.abs(gradx) + np.abs(grady)
T = np.arctan(grady / (gradx + 1e-3))

# 3. 细化边缘
h, w = R.shape
img_thin = np.zeros_like(R)
for i in range(1, h - 1):
    for j in range(1, w - 1):
        theta = T[i, j]
        if -np.pi / 8 < theta < np.pi / 8:
            if R[i, j] == max([R[i, j], R[i, j - 1], R[i, j + 1]]):
                img_thin[i, j] = R[i, j]
        elif -3 * np.pi / 8 < theta < -np.pi / 8:
            if R[i, j] == max([R[i, j], R[i - 1, j + 1], R[i + 1, j - 1]]):
                img_thin[i, j] = R[i, j]
        elif np.pi / 8 < theta < 3 * np.pi / 8:
            if R[i, j] == max([R[i, j], R[i - 1, j - 1], R[i + 1, j + 1]]):
                img_thin[i, j] = R[i, j]
        else:
            if R[i, j] == max([R[i, j], R[i - 1, j], R[i + 1, j]]):
                img_thin[i, j] = R[i, j]

cv.imshow('xihuabianyuan', img_thin)
cv.waitKey(0)

# 设置阈值
th1 = 20
th2 = 200
h, w = img_thin.shape
img_edge = np.zeros_like(img_thin, dtype=np.uint8)
for i in range(1, h - 1):
    for j in range(1, w - 1):
        if img_thin[i, j] > th2:
            img_edge[i, j] = img_thin[i, j]
        elif img_thin[i, j] > th1:
            around = img_thin[i - 1: i + 2, j - 1: j + 2]
            if around.max() > th2:
                img_edge[i, j] = img_thin[i, j]

cv.imshow('result', img_edge)
cv.waitKey(0)
cv.destroyAllWindows()





import cv2 as cv
import numpy as np

# 读入灰度图像
img = cv.imread('shuzi\cat_gray.jpg', cv.IMREAD_GRAYSCALE)

# 应用霍夫变换进行线检测
edges = cv.Canny(img, 50, 150, apertureSize=3)
lines = cv.HoughLines(edges, 1, np.pi / 180, threshold=100)

# 绘制检测到的线
if lines is not None:
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)


# 显示结果
cv.imshow('Line Detection Result', img)
cv.waitKey(0)
cv.destroyAllWindows()


import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# 1. 将图片以灰度的方式读取进来
img = cv.imread("shuzi\pingguo.jpg", cv.IMREAD_COLOR)
cv.imshow("src",img)

gray_img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# cv.imshow("gray",gray_img)
#
flag,thresh_img = cv.threshold(gray_img,100,255,cv.THRESH_BINARY_INV)
cv.imshow("thresh_img",thresh_img)


# 3. 霍夫变换
#  线段以像素为单位的距离精度，double类型的，推荐用1.0
rho = 1
# 线段以弧度为单位的角度精度，推荐用numpy.pi/180
theta = np.pi/180
# 累加平面的阈值参数，int类型，超过设定阈值才被检测出线段，值越大，基本上意味着检出的线段越长，检出的线段个数越少。
threshold=10
# 线段以像素为单位的最小长度
min_line_length=25
# 同一方向上两条线段判定为一条线段的最大允许间隔（断裂），超过了设定值，则把两条线段当成一条线段，值越大，允许线段上的断裂越大，越有可能检出潜在的直线段
max_line_gap = 3

lines = cv.HoughLinesP(thresh_img,rho,theta,threshold,minLineLength=min_line_length,maxLineGap=max_line_gap)

dst_img = img.copy()

for line in lines:
    print(lines)
    x1,y1,x2,y2 = line[0]
    cv.line(dst_img,(x1,y1),(x2,y2),(0,0,255),2)

cv.imshow("dst img",dst_img)

cv.waitKey(0)
cv.destroyAllWindows()




#7
import cv2 as cv
import matplotlib.pyplot as plt

# 读取灰度图像
img = cv.imread('shuzi\pingguo.jpg', 0)

# 显示图像
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.show()

# 绘制直方图
plt.hist(img.ravel(), 256, [0, 256])
plt.show()

# 二值化图像（反转）
_, img_bin = cv.threshold(img, 125, 255, cv.THRESH_BINARY_INV)

# 显示二值化图像
plt.imshow(img_bin, cmap='gray')
plt.axis('off')
plt.show()


#8

import cv2 as cv
import numpy as np

# 读取灰度图像
img = cv.imread('shuzi\pingguo.jpg', 0)

# 初始化阈值
threshold = 128

# 循环迭代直到阈值不再变化
while True:
    # 划分像素为背景和目标
    bg = img[img < threshold]
    obj = img[img >= threshold]

    # 计算背景和目标的均值
    bg_mean = np.mean(bg)
    obj_mean = np.mean(obj)

    # 更新阈值为新的均值
    new_threshold = int((bg_mean + obj_mean) / 2)

    # 判断阈值是否收敛
    if new_threshold == threshold:
        break

    # 更新阈值
    threshold = new_threshold

# 根据最终阈值进行图像分割
binary_img = np.zeros_like(img)
binary_img[img >= threshold] = 255

# 显示分割结果
cv.imshow('Segmented Image', binary_img)
cv.waitKey(0)
cv.destroyAllWindows()

#9
import cv2 as cv

# 读取灰度图像
img = cv.imread('shuzi\pingguo.jpg', 0)

# 初始化最佳阈值和最佳方差
best_threshold = 0
best_sigma = -1

# 计算背景和目标像素数量
bg_size = 0
obj_size = img.size

# 计算背景和目标像素的均值（初始为0）
bg_mean = 0
obj_mean = img.mean()

# 循环遍历阈值的取值范围
for t in range(0, 256):
    # 根据阈值对图像进行分割，得到背景和目标像素
    bg = img[img < t]
    obj = img[img >= t]

    # 更新背景和目标像素的数量
    bg_size = bg.size
    obj_size = obj.size

    # 计算背景和目标像素的均值
    bg_mean = 0 if bg_size == 0 else bg.mean()
    obj_mean = 0 if obj_size == 0 else obj.mean()

    # 计算方差
    sigma = bg_size * obj_size * (bg_mean - obj_mean) ** 2

    # 如果方差大于之前的最佳方差，则更新最佳方差和最佳阈值
    if sigma > best_sigma:
        best_sigma = sigma
        best_threshold = t

# 打印最佳阈值
print(f"Best threshold: {best_threshold}")
from collections import deque
from pathlib import Path
from tkinter import *
import timeit

import cv2
import numpy as np
from PIL import Image, ImageTk
from scipy import ndimage
from skimage import io


class Canny:
    def __init__(self, sigma=1.4, kernel=5, interactive=True, thresh_low=30, thresh_high=200):
        self.image_folder = Path('images')
        self.sigma = sigma
        assert kernel // 2, f'kernel size must be an odd number, but get {kernel}'
        self.kernel = kernel
        self.interactive = interactive
        self.thresh_low = thresh_low
        self.thresh_high = thresh_high
        self.origin_image = None
        self.max_edge = 512
        self.save_path = Path('../dip_report/image')

    def process(self, image='base.jpg'):
        image = self.read_image(image)
        start = timeit.default_timer()
        image = self.smooth(image)
        dx, dy, magnitude = self.gradient(image)
        nms = self.nms(magnitude, dx, dy)
        if not self.interactive:
            thresh = self.double_threshold(nms)
            end = timeit.default_timer()
            print(f'time used: {end-start}')
            self.show(thresh)
        else:
            self.interact(nms)

    def interact(self, nms):
        root = Tk()
        image = Image.fromarray(self.origin_image)
        image = ImageTk.PhotoImage(image=image)
        s3 = Label(root, image=image)
        s3.pack()
        s1 = Scale(root, from_=0, to=255, orient=HORIZONTAL, resolution=1,
                   tickinterval=10, length=600)
        s1.set(self.thresh_low)
        s1.pack()
        s2 = Scale(root,
                   from_=0,  # 设置最小值
                   to=255,  # 设置最大值
                   orient=HORIZONTAL,  # 设置横向
                   resolution=1,  # 设置步长
                   tickinterval=10,  # 设置刻度
                   length=600,  # 设置像素
                   )  # 绑定变量
        s2.set(self.thresh_high)
        s2.pack()

        def show():
            if s1.get() >= s2.get():
                self.thresh_high = s1.get()
                self.thresh_low = s2.get()
            else:
                self.thresh_high = s2.get()
                self.thresh_low = s1.get()
            new_thresh = self.double_threshold(nms)
            new_img = Image.fromarray(new_thresh)
            new_img = ImageTk.PhotoImage(new_img)
            s3.configure(image=new_img)
            s3.image = new_img

        Button(root, text='更新阈值并检测边界', command=show).pack()  # 用command回调函数获取位置
        root.mainloop()

    def smooth(self, image):
        gaussian_kernel = np.zeros([self.kernel, self.kernel])
        kernel = self.kernel
        for i in range(kernel):
            for j in range(kernel):
                gaussian_kernel[i][j] = np.exp(-((i - kernel // 2) ** 2 + (j - kernel // 2) ** 2) / 2 / self.sigma ** 2)
        gaussian_kernel /= gaussian_kernel.sum()
        height, width = image.shape
        blurred = image.copy()
        for h in range(height - kernel):
            for w in range(width - kernel):
                blurred[h + kernel // 2, w + kernel // 2] = np.sum(image[h:h + kernel, w:w + kernel] * gaussian_kernel)

        return blurred

    def hist(self, image):
        new_image = np.zeros_like(image)
        image = image.astype(np.uint8)
        height, width = image.shape
        count = np.zeros(256)
        s = np.zeros(256, dtype=new_image.dtype)
        for i in range(height):
            for j in range(width):
                count[image[i][j]] += 1
        sigma = 0
        zero_count = count[0]
        count[0] = 0
        for i in range(256):
            sigma += count[i]
            s[i] = 255 * sigma / max(height * width - zero_count, 1)
        for i in range(height):
            for j in range(width):
                new_image[i][j] = s[image[i][j]]
        return new_image

    def gradient(self, image):
        image = image.astype(np.float32)
        kernel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)
        kernel_y = kernel_x.transpose()
        dx = ndimage.filters.convolve(image, kernel_x)
        dy = ndimage.filters.convolve(image, kernel_y)
        # dx = np.zeros_like(image)
        # dy = np.zeros_like(image)
        # height, width = image.shape
        # dy[:height - 1] = image[:height - 1] - image[1:]
        # dx[:, :width - 1] = image[:, :width - 1] - image[:, 1:]

        magnitude = np.sqrt(dx ** 2 + dy ** 2)
        magnitude = magnitude / magnitude.max() * 255
        return dx, dy, magnitude

    def nms(self, magnitude, dx, dy):
        height, width = magnitude.shape
        nms = np.copy(magnitude)

        for h in range(1, height - 1):
            for w in range(1, width - 1):
                if (self.interactive and magnitude[h, w] > 0) or (
                        not self.interactive and magnitude[h, w] > self.thresh_low):
                    grad_x = dx[h, w]  # 当前点 x 方向导数
                    grad_y = dy[h, w]  # 当前点 y 方向导数
                    grad = magnitude[h, w]  # 当前梯度点
                    direction = int(grad_x * grad_y > 0) * 2 - 1
                    if np.abs(grad_y) > np.abs(grad_x):
                        weight = abs(grad_x) / max(abs(grad_y), 1e-6)
                        grad2 = magnitude[h - 1, w]
                        grad4 = magnitude[h + 1, w]
                        grad1 = magnitude[h - 1, w - direction]
                        grad3 = magnitude[h + 1, w + direction]
                    else:
                        weight = abs(grad_y) / max(abs(grad_x), 1e-6)
                        grad2 = magnitude[h, w - 1]
                        grad4 = magnitude[h, w + 1]
                        grad1 = magnitude[h - direction, w - 1]
                        grad3 = magnitude[h + direction, w + 1]

                    grad_temp1 = (1 - weight) * grad1 + weight * grad2
                    grad_temp2 = (1 - weight) * grad3 + weight * grad4
                    if grad <= grad_temp1 or grad < grad_temp2:
                        nms[h, w] = 0

        nms[0, :] = nms[height - 1, :] = 0
        nms[:, 0] = nms[:, width - 1] = 0
        return nms

    def check_iterative(self, boundary, nms, checked, lists):

        stack = deque()
        if not len(lists[0]):
            return
        for row, column in zip(*lists):
            stack.append((row, column))
        while stack:
            i, j = stack.popleft()
            if i < 0 or j < 0 or i >= boundary.shape[0] or j >= boundary.shape[1]:
                continue
            if checked[i][j]:
                continue
            checked[i][j] = 1
            if self.thresh_low >= nms[i][j]:
                continue
            boundary[i][j] = 1
            for row in (-1, 0, 1):
                for column in (-1, 0, 1):
                    if row or column:
                        stack.append((i + column, j + row))

    def double_threshold(self, nms):
        new_nms = nms.copy()
        new_nms[new_nms < self.thresh_low] = 0
        new_nms = self.hist(new_nms)
        boundary = np.zeros_like(new_nms)
        boundary[new_nms >= self.thresh_high] = 1
        checked = np.zeros_like(new_nms)
        self.check_iterative(boundary, new_nms, checked, np.where(new_nms > self.thresh_high))
        return (boundary * 255).astype(np.uint8)

    def read_image(self, image):
        image = self.image_folder / image
        image = io.imread(image)
        if max(image.shape) > self.max_edge and self.interactive:
            shape = image.shape[:2]
            new_shape = np.array(shape) / max(shape) * self.max_edge
            image = cv2.resize(image, tuple(new_shape[::-1].astype(np.int)))
        self.origin_image = image
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return image

    def show(self, image):
        # image = image.astype(np.uint8)
        cv2.imshow("Canny", image)
        # cv2.imshow("opencv", cv2.Canny(self.origin_image, 50, 200))
        # cv2.imwrite("../dip_report/image/canny_mine.png", image)
        cv2.waitKey(0)


if __name__ == '__main__':
    Canny(1.4, 5, False, 13, 100).process('lenna.png')

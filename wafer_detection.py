# _*_ coding : utf-8 _*_ 
# @Time : 2022/6/17 17:42
# @Author : 高德傲
# @File : 轮廓匹配
# @Project : opencv_test
import time
from collections import Counter

import cv2
import numpy as np


def py_nms(dets, thresh):
    """Pure Python NMS baseline."""
    # x1、y1、x2、y2、以及score赋值
    # （x1、y1）（x2、y2）为box的左上和右下角标
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    # 每一个候选框的面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # order是按照score降序排序的
    order = scores.argsort()[::-1]
    # print("order:",order)

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        # 计算当前概率最大矩形框与其他矩形框的相交框的坐标，会用到numpy的broadcast机制，得到的是向量
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        # 计算相交框的面积,注意矩形框不相交时w或h算出来会是负数，用0代替
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        # 计算重叠度IOU：重叠面积/（面积1+面积2-重叠面积）
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        # 找到重叠度不高于阈值的矩形框索引
        inds = np.where(ovr <= thresh)[0]
        # print("inds:",inds)
        # 将order序列更新，由于前面得到的矩形框索引要比矩形框在原order序列中的索引小1，所以要把这个1加回来
        order = order[inds + 1]
    return keep


def template(img_gray, template_img, template_threshold):
    '''
    img_gray:待检测的灰度图片格式
    template_img:模板小图，也是灰度化了
    template_threshold:模板匹配的置信度
    '''

    h, w = template_img.shape[:2]
    res = cv2.matchTemplate(img_gray, template_img, cv2.TM_CCOEFF_NORMED)
    start_time = time.time()
    loc = np.where(res >= template_threshold)  # 大于模板阈值的目标坐标
    score = res[res >= template_threshold]  # 大于模板阈值的目标置信度
    # 将模板数据坐标进行处理成左上角、右下角的格式
    xmin = np.array(loc[1])
    ymin = np.array(loc[0])
    xmax = xmin + w
    ymax = ymin + h
    xmin = xmin.reshape(-1, 1)  # 变成n行1列维度
    xmax = xmax.reshape(-1, 1)  # 变成n行1列维度
    ymax = ymax.reshape(-1, 1)  # 变成n行1列维度
    ymin = ymin.reshape(-1, 1)  # 变成n行1列维度
    score = score.reshape(-1, 1)  # 变成n行1列维度
    data_hlist = []
    data_hlist.append(xmin)
    data_hlist.append(ymin)
    data_hlist.append(xmax)
    data_hlist.append(ymax)
    data_hlist.append(score)
    data_hstack = np.hstack(data_hlist)  # 将xmin、ymin、xmax、yamx、scores按照列进行拼接
    thresh = 0.3  # NMS里面的IOU交互比阈值

    keep_dets = py_nms(data_hstack, thresh)
    print("time:", time.time() - start_time)  # 打印数据处理到nms运行时间
    dets = data_hstack[keep_dets]  # 最终的nms获得的矩形框
    return dets


def canny_edges(img, threshold1, threshold2):
    # Convert to graycsale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Blur the image for better edge detection
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
    res = cv2.Canny(img_blur, threshold1, threshold2)
    return res


# Optimize rectangular contour
def zeyou(ct):
    ff = list(ct)
    l = len(ff)
    a = np.array(ff)
    a = a.reshape(l, 2)
    # .T为矩阵转置
    b = a[:, 0].T  # 横坐标
    c = a[:, 1].T  # 纵坐标
    L = len(b.T)
    S = len(c.T)

    b = np.array(b, dtype=str)
    b = b.reshape(L)
    b = b.tolist()

    c = np.array(c, dtype=str)
    c = c.reshape(S)
    c = c.tolist()

    # print(Counter(b))
    # print(Counter(c))
    result_x = dict(Counter(b).most_common(4))
    # zx = zip(result_x.keys())
    zx = list(result_x.keys())
    zx1 = sorted(list(map(int, zx)))
    zx2 = list(map(str, zx1))
    result_y = dict(Counter(c).most_common(4))
    zy = list(result_y.keys())
    zy1 = sorted(list(map(int, zy)))
    zy2 = list(map(str, zy1))
    # print(result_x)
    # print(result_y)
    # 分别选出横、纵坐标出现次数最多和次多的两个值,l--左，r--右，d--下，u--上，ll--左备选,类推
    x_l = zx2[0]
    x_r = zx2[3]
    x_l_l = zx2[1]
    x_r_r = zx2[2]
    y_d = zy2[3]
    y_u = zy2[0]
    y_d_d = zy2[2]
    y_u_u = zy2[1]
    # 对主选备选值进行择优
    if int(x_l) > int(x_r):
        Zx = x_l
        x_l = x_r
        x_r = Zx
        if 1 <= int(x_l_l) - int(x_l) <= 5:
            x_l = x_l_l
        else:
            if -5 <= int(x_l_l) - int(x_r) <= -1:
                x_r = x_l_l

        if -5 <= int(x_r_r) - int(x_r) <= -1:
            x_r = x_r_r
        else:
            if 1 <= int(x_r_r) - int(x_l) <= 5:
                x_l = x_r_r

    if int(y_u) > int(y_d):
        Zy = y_u
        y_u = y_d
        y_d = Zy
        if -5 <= int(y_d_d) - int(y_d) <= -1:
            y_d = y_d_d
        else:
            if 1 <= int(y_d_d) - int(y_u) <= 5:
                y_u = y_d_d

        if 1 <= int(y_u_u) - int(y_u) <= 5:
            y_u = y_u_u
        else:
            if -5 <= int(y_u_u) - int(y_d) <= -1:
                y_d = y_u_u

    if 1 <= int(x_l_l) - int(x_l) <= 5:
        x_l = x_l_l
    else:
        if -5 <= int(x_l_l) - int(x_r) <= -1:
            x_r = x_l_l

    if -5 <= int(x_r_r) - int(x_r) <= -1:
        x_r = x_r_r
    else:
        if 1 <= int(x_r_r) - int(x_l) <= 5:
            x_l = x_r_r

    if int(x_l) > int(x_r):
        Zx = x_l
        x_l = x_r
        x_r = Zx

    if -5 <= int(y_d_d) - int(y_d) <= -1:
        y_d = y_d_d
    else:
        if 1 <= int(y_d_d) - int(y_u) < 5:
            y_u = y_d_d

    if 1 <= int(y_u_u) - int(y_u) <= 5:
        y_u = y_u_u
    else:
        if -5 <= int(y_u_u) - int(y_d) <= -1:
            y_d = y_u_u

    if int(y_u) > int(y_d):
        Zy = y_u
        y_u = y_d
        y_d = Zy

    # print(x_l, x_r, y_u, y_d)
    # 找出横纵坐标出现次数最多和次多的两个值所在的坐标，并转换成三维数组
    list_x = []
    for p in range(l):
        if x_l == b[p] or x_r == b[p] or y_d == c[p] or y_u == c[p]:
            list_x.append(a[p])
    ct = np.array(list_x).reshape(len(list_x), 1, 2)
    return ct, x_r, x_l, y_u, y_d


def zuobiaopaixu(zuobiao):
    liebiao = []
    lenth = len(zuobiao)
    for i in range(lenth):
        for t in range(lenth - i - 1):
            if zuobiao[t][0] > zuobiao[t + 1][0] and zuobiao[t][0] - zuobiao[t + 1][0] > 30:
                zuobiao[t], zuobiao[t + 1] = zuobiao[t + 1], zuobiao[t]
            if zuobiao[t][1] > zuobiao[t + 1][1] and zuobiao[t][1] - zuobiao[t + 1][1] > 30:
                zuobiao[t], zuobiao[t + 1] = zuobiao[t + 1], zuobiao[t]
    for h in range(lenth):
        liebiao.append(zuobiao[h])
    return liebiao


# Select largest Contour
def SelectCountours(contours):
    global cnt
    if len(contours) > 1:
        m = len(contours[0])
        n = 0
        for j in range(len(contours)):
            if len(contours[j]) > m:
                m = len(contours[j])
                n = j
                cnt = contours[n]
            else:
                cnt = contours[n]
    else:
        cnt = contours[0]
    return cnt


def nothing(x):
    pass


def Make_Muban(img_ori):
    # img_ori = cv2.imread(ori_path)  # original image
    img_fil = cv2.medianBlur(img_ori, 3)  # 中值滤波
    # 设置卷积核
    kernel = np.ones((5, 5), np.uint8)
    # 闭运算
    img = cv2.morphologyEx(img_fil, cv2.MORPH_CLOSE, kernel)
    cv2.namedWindow('roi_pre', cv2.WINDOW_FREERATIO)
    # 鼠标选取模板
    roi_pre = cv2.selectROI(windowName="roi_pre", img=img, showCrosshair=True, fromCenter=False)
    img_roi = img[int(roi_pre[1]):int(roi_pre[1] + roi_pre[3]), int(roi_pre[0]):int(roi_pre[0] + roi_pre[2])]
    # 滑动栏改变阈值
    cv2.namedWindow('Canny Edge Detection of ROI', flags=0)
    cv2.createTrackbar('threshold1', "Canny Edge Detection of ROI", 0, 255, nothing)
    cv2.createTrackbar('threshold2', "Canny Edge Detection of ROI", 0, 255, nothing)
    cv2.setTrackbarPos('threshold1', "Canny Edge Detection of ROI", 50)
    cv2.setTrackbarPos('threshold2', "Canny Edge Detection of ROI", 120)
    while True:
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        minVal = cv2.getTrackbarPos('threshold1', 'Canny Edge Detection of ROI')
        maxVal = cv2.getTrackbarPos('threshold2', 'Canny Edge Detection of ROI')
        roi_cannyed = canny_edges(img_roi, minVal, maxVal)
        # result = np.hstack([edges, threshold1, threshold2])
        cv2.imshow('Canny Edge Detection of ROI', roi_cannyed)

    # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
    contours, hierarchy = cv2.findContours(image=roi_cannyed, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)

    # draw contours on the original image
    image_copy = img_roi.copy()
    # cv2.imwrite('img_roi.jpg', image_copy)
    res = cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=1,
                           lineType=cv2.LINE_AA)
    cv2.imshow('canny', res)
    # cv2.waitKey(0)
    cnt = SelectCountours(contours)
    # rect = cv2.minAreaRect(cnt)
    cv2.waitKey(0)
    '''最小外接矩形择优算法'''
    cnt, x_r, x_l, y_u, y_d = zeyou(cnt)
    cv2.imwrite('muban Contours.jpg', image_copy)
    cv2.destroyAllWindows()
    res4 = cv2.rectangle(res, (int(x_l), int(y_u)), (int(x_r), int(y_d)), (0, 0, 255), 1)  # res4是优化过的矩形轮廓

    # 截取模板
    Muban = img_roi[int(y_u):int(y_d), int(x_l):int(x_r)]
    cv2.imwrite(address_muban, Muban)
    # img_gray4 = cv2.cvtColor(res4, cv2.COLOR_BGR2GRAY)  # 转化成灰度图
    cv2.imwrite('img_gray_4.jpg', res4)
    cv2.imshow('rectangle', res4)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # print(res4)
    return Muban, cnt, img


def Template_result(img, dets):
    coo = []
    dy = 30  # puttext中换行参数
    Mask_rectangle = np.zeros(img.shape[0:2], dtype="uint8")
    for coord in dets:
        cv2.rectangle(img_ori, (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3])), (0, 0, 255),
                      2)
        cv2.rectangle(Mask_rectangle, (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3])), 255, -1)

        X = int((int(coord[0]) + int(coord[2])) / 2)  # 中心点横坐标
        Y = int((int(coord[1]) + int(coord[3])) / 2)  # 中心点纵坐标
        cv2.circle(img_ori, (X, Y), 5, (0, 0, 0), -1)
        coo.append((X, Y))
        '''2022.9.21 匹配得到的X、Y值以坐标形式放入数组中，对X、Y进行排序，
        按大小重新赋值（1，2，3，...）（问题：同一行或同一列有坐标相差较小的坐标值存在，需视做同一行或同一列，即赋相同的值）
        cv2.putText(img_ori)'''
    cv2.namedWindow("Result", 0)
    cv2.imshow('Result', img_ori)
    cv2.waitKey(0)
    mask_inv = cv2.bitwise_not(Mask_rectangle)  # 非运算，mask取反
    masked = cv2.bitwise_and(img_ori, img_ori, mask=mask_inv)
    ret, thresh2 = cv2.threshold(masked, 50, 220, cv2.THRESH_BINARY_INV)
    # 设置卷积核
    kernel = np.ones((4, 4), np.uint8)
    # 膨胀
    thresh2_dilate = cv2.dilate(thresh2, kernel)
    masked_dilate = cv2.dilate(masked, kernel)
    imgAddCV = cv2.add(masked_dilate, thresh2_dilate)  # OpenCV 加法: 饱和运算
    imgAddCV_gray = cv2.cvtColor(imgAddCV, cv2.COLOR_BGR2GRAY)
    template_threshold = 0.6
    dets2 = template(imgAddCV_gray, roi, template_threshold)
    for coords in dets2:
        cv2.rectangle(img_ori, (int(coords[0]), int(coords[1])), (int(coords[2]), int(coords[3])), (0, 255, 0),
                      2)
        X = int((int(coords[0]) + int(coords[2])) / 2)
        Y = int((int(coords[1]) + int(coords[3])) / 2)
        cv2.circle(img_ori, (X, Y), 5, (0, 255, 0), -1)
        coo.append((X, Y))
    abc = zuobiaopaixu(coo)
    coo_list = [int(p) for p in range(len(abc))]
    num_x = 1
    num_y = 1
    coo_list[0] = (1, 1)
    '''
    img—需要添加文本内容的图像。
    text—具体要添加的文本内容，为字符串类型。
    org—要添加文本内容的左上角坐标位置。
    fontFace—字体类型，大家用得比较多的是字体FONT_HERSHEY_SIMPLEX。
    fontScale—字体缩放尺度，实际上就是控制内容的大小。
    color—字体颜色。
    thickness—字体粗细。
    lineType—字体线条类型。
    bottomLeftOrigin—图像坐标原点位置是否位于左下角，当这个值为true时，图像坐标原点位置位于左下角，当这个值为false时，图像坐标原点位置位于左上角。
    '''
    dets = np.vstack((dets, dets2))  # 拼接结果
    test = str(coo_list[0]) + "\n" + str('%.3f' % (dets[0, 4]))
    for i, txt in enumerate(test.split('\n')):
        cv2.putText(img_ori, txt, (abc[0][0], abc[0][1] + dy * i), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0, 0, 255), 3,
                    bottomLeftOrigin=False)
    for i in range(len(abc) - 1):
        if abc[i + 1][0] > abc[i][0]:
            num_y += 1
            coo_list[i + 1] = (num_x, num_y)
            test1 = str(coo_list[i + 1]) + "\n" + str('%.3f' % (dets[i + 1, 4]))
            for j, txt in enumerate(test1.split('\n')):
                cv2.putText(img_ori, txt, (abc[i + 1][0], abc[i + 1][1] + dy * j),
                            cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0, 0, 255), 3,
                            bottomLeftOrigin=False)
        else:
            num_x += 1
            num_y = 1
            coo_list[i + 1] = (num_x, num_y)
            test2 = str(coo_list[i + 1]) + "\n" + str("%.3f" % (dets[i + 1, 4]))
            for j, txt in enumerate(test2.split('\n')):
                cv2.putText(img_ori, txt, (abc[i + 1][0], abc[i + 1][1] + dy * j),
                            cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0, 0, 255), 3,
                            bottomLeftOrigin=False)
    print(coo_list)
    cv2.putText(img_ori, txt, (abc[i + 1][0], abc[i + 1][1] + dy * j),
                cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0, 0, 255), 3,
                bottomLeftOrigin=False)
    cv2.imwrite(address_result, img_ori)
    cv2.namedWindow("final", 0)
    cv2.imshow("final", img_ori)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


address_image = "C:\\Users\\GDA\\Desktop\\wafer image\\butongbeilv\\dan_kong1/3b1c.bmp"  # 待测图片储存位置
address_muban = 'C:/Users/GDA/Desktop/chips/muban_dan_kong1_1b1c.png'  # 模板图片储存位置（根据鼠标选定的区域）
address_result = "dan_kong1/result_1b1c.jpg"  # 匹配结果保存位置

if __name__ == "__main__":
    img_ori = cv2.imread(address_image)  # 待测图片
    muban, cnt, img = Make_Muban(img_ori)  # 制作模板
    roi = cv2.cvtColor(muban, cv2.COLOR_BGR2GRAY)  # 将模板图转为灰度图
    template_threshold = 0.6  # 模板置信度
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将原图转为灰度图
    dets = template(img_gray, roi, template_threshold)  # 模板匹配
    Template_result(img, dets)  # 展示匹配结果

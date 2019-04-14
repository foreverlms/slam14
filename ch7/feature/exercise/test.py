#!/usr/bin/env python
#coding:utf-8
import cv2
import numpy as np

print(cv2.__version__)
cv2.namedWindow("gray")
img = np.zeros((512,512),np.uint8)#生成一张空的灰度图像
cv2.line(img,(0,0),(511,511),255,5)#绘制一条白色直线
cv2.imshow("gray",img)#显示图像
#循环等待，按q键退出
while True:
	key=cv2.waitKey(1)
	if key==ord("q"):
		break
cv2.destoryWindow("gray")

# str_source = raw_input()

# list_ = str_source.split(" ");

# print list_

# N = int(list_[0]);

# interval_list = []

# for i in range(1,len(list_)):
#     if len(list_[i]) == 0:
#         continue;
#     j = 0;
#     while len(list_[i]) - 8*(j+1) >= 0:
#         tmp = list_[i][j:8*(j+1)]
#         interval_list.append(tmp);
#         j+=1;
#     # tmp_str = list_[i][8*j:]+"0"*(8-len(list_[i][8*j:]))
#     tmp_str = list_[i][8*j:]
#     interval_list.append(tmp_str)

# interval_list = sorted(interval_list) 


# for i in range(0,len(interval_list)):
#     if len(interval_list[i]) < 8:
#         print interval_list[i] + "0" * (8-len(interval_list[i])),
#     else:
#         print interval_list[i],

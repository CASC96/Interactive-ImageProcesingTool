import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def get_overlapping_outlines(outlines1, outlines2):
    #outlines1 = np.array(outlines1,dtype=np.object_)
    
    #outlines2 = np.array(outlines2,dtype=np.object_)
    overlapping_outlines = []
    for i, outline1 in enumerate(outlines1):
        overlaps = []
        for pixel1 in outline1:
            # Check if the pixel overlaps with any pixel in the second list
            overlaps_xy = np.isin(outlines2, pixel1)
            # Check if the pixel overlaps with any of the boxes surrounding it
            boxes = [(pixel1[0] - 1, pixel1[1]), (pixel1[0] + 1, pixel1[1]), (pixel1[0], pixel1[1] - 1), (pixel1[0], pixel1[1] + 1), (pixel1[0] - 1, pixel1[1] - 1), (pixel1[0] + 1, pixel1[1] - 1), (pixel1[0] - 1, pixel1[1] + 1), (pixel1[0] + 1, pixel1[1] + 1)]
            overlaps_boxes = np.isin(outlines2, boxes)
            # Check if there are two or more overlapped pixels
            if np.sum(overlaps_xy) + np.sum(overlaps_boxes) >= 2:
                overlaps.append(True)
        # Check if there are two or more overlapped pixels in the outline
        if sum(overlaps) >= 2:
            overlapping_outlines.append(i)
    return overlapping_outlines


def get_intersecting_rectangles_v2(rectangles1, rectangles_list):
    rectangles1 = np.array(rectangles1)
    rectangles_lista = np.array(rectangles_list)
    burbujas_generadas=[]
    burbujas_consumidas=[]
    image = np.zeros((1080,1920, 3), np.uint8)
    #cv.namedWindow('dx', cv.WINDOW_GUI_NORMAL)

    x2, y2, w2, h2 = rectangles_list.T


    for i, rectangles in enumerate(rectangles1):
        x1, y1, w1, h1 = rectangles.T
        # Select the rectangles that intersect
        a = np.logical_and((x2 < x1 + w1), True)
        b = np.logical_and((x2 + w2 > x1), True)
        c = np.logical_and((y2 < y1 + h1), True)
        d = np.logical_and((y2 + h2 > y1), True)
        
        e=np.bitwise_and(a,b)
        f=np.bitwise_and(c,d)
        g=np.bitwise_and(e,f)
        
        
        
        result = np.argwhere(g == True)
       
        
        # Get the indices of the intersecting rectangles
        intersection_indices = len(result)
        # Check if there are at least two intersecting rectangles
        if intersection_indices >= 2:
            burbujas_generadas.append(np.array(i,dtype=np.int16))
            burbujas_consumidas.append(np.array(result,dtype=np.object_))
            
            cv.rectangle(image, (x1, y1), (x1 + w1, y1 + h1), (255, 0, 0), 2)
            
            evento = len(burbujas_generadas)-1
            for n in range(intersection_indices):
                rect_cons=rectangles_lista[list(burbujas_consumidas[evento][n])[0]]
                x22, y22, w22, h22 = rect_cons.T
                cv.rectangle(image, (x22, y22), (x22 + w22, y22 + h22), (0, 255, 0), 2)
        
        
            cv.imshow('Image', image)
            cv.waitKey(30)  
            intersection_indices=0
                
    return (burbujas_generadas, burbujas_consumidas)


A =np.load('CI_array_totalFrame.npy',allow_pickle=True)
Areas=np.load('AF_array_totalFrame.npy',allow_pickle=True)
BR=np.load('BR_array_totalFrame.npy',allow_pickle=True)


for frame in range(99):
    bubbles=get_intersecting_rectangles_v2(BR[frame+1], BR[frame])

first_list = np.asarray([lst[0] for lst in bubbles])
second_list = np.asarray([lst[1] for lst in bubbles],dtype=np.object_)



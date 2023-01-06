import cv2 as cv
import numpy as np
import imcropcs


def Video_Crop(ncarpetas, nvideos):
    videos = np.zeros((ncarpetas[0], nvideos[0]), dtype=np.object0)
    for nc in range(ncarpetas[0]):
        for nv in range(nvideos[0]):
            videos[nc][nv] = (str(str(nc+1)+"/"+str(nv+1)+".MP4"))

    croppData = np.zeros((ncarpetas[0], nvideos[0]), dtype=np.object0)
    for nc in range(ncarpetas[0]):
        for nv in range(nvideos[0]):

            video = videos[nc, nv]

            Data_video = cv.VideoCapture(video)
            success, first_image = Data_video.read()
            """     while True:
                success, img = Data_video.read()
                if success:
                    print(img)#getting None
                    print(success)#getting False """

            # ingresar el primer fotograma del video
            # first_image = 'test.jpg'

            # data_first_image = cv.imread(first_image)
            x_start, y_start, x_end, y_end = imcropcs.cropcs(first_image)

            # print(x_start, x_end, y_start, y_end)
            if x_start > x_end:
                x_start, x_end = x_end, x_start
            if y_start > y_end:
                y_start, y_end = y_end, y_start
            # print(x_start, x_end, y_start, y_end)

            # cropped_first_image = first_image[y_start:y_end, x_start:x_end]

            croppy = np.array([x_start, y_start, x_end, y_end])

            croppData[nc][nv] = croppy

    return (croppData, videos)


def Video_Threshold_Debug(video_folder, cropArea, ncarpeta, nvideo, low_H, high_H, low_S, high_S, low_V, high_V):

    cv.namedWindow("image", cv.WINDOW_GUI_NORMAL)
    cv.setWindowProperty("image", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
    video = cv.VideoCapture(video_folder)

    TOTAL_FRAME = video.get(cv.CAP_PROP_FRAME_COUNT)
    TOTAL_FRAME = int(TOTAL_FRAME)

    # backSub = cv.createBackgroundSubtractorMOG2(-1, -1, False)

    CI_array_totalFrame = np.ndarray(
        TOTAL_FRAME, dtype=np.object0)

    CT_array_totalFrame = np.ndarray(
        TOTAL_FRAME, dtype=np.object0)

    for f in range(100):  # range(TOTAL_FRAME):

        video.set(cv.CAP_PROP_POS_FRAMES, f)
        ret, frame = video.read()

        frame = frame[cropArea[1]:cropArea[3], cropArea[0]:cropArea[2]]
        cv.imshow('image', frame)
        cv.waitKey(1)
        gray_image = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame_HSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        # threshold_image = backSub.apply(frame)

        ret2, th0 = cv.threshold(
            gray_image, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

        blur = cv.bilateralFilter(gray_image, 10, 10, 10)
        # Otsu's thresholding after Gaussian filtering
        ret2, th1 = cv.threshold(
            blur, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

        th2 = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_MEAN_C,
                                   cv.THRESH_BINARY_INV, 11, 2)

        th3 = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv.THRESH_BINARY_INV, 11, 2)

        # threshold_image_1 = cv.bitwise_not(threshold_image_1)
        threshold_image_1 = cv.bitwise_or(th2, th3)
        threshold_image_2 = cv.bitwise_or(th0, th1)

        # SImple thresholding
        # 122 es el valor arbitrario
        # ret, th001 = cv.threshold(gray_image, 122, 255, cv.THRESH_BINARY_INV)

        # esta imagen binaria es calculada mediante metodos adaptativos
        threshold_image_n = cv.bitwise_or(threshold_image_1, threshold_image_2)

        frame_threshold_HSV = cv.inRange(
            frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))
        frame_threshold_HSV = cv.bitwise_not(frame_threshold_HSV)

        threshold_image = cv.bitwise_or(threshold_image_n, frame_threshold_HSV)

        # CI_Mask = cv.bitwise_not(threshold_image)

        #  contornos
        # contours = np.zeros((1, 1), dtype=np.object0)

        contornos_0, hierarchy = cv.findContours(
            threshold_image, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        CI_area_array = np.ndarray(
            (len(contornos_0)), dtype=np.int32)
        # filtro por area
        for cont in range(len(contornos_0)):
            CI_area_array[cont] = cv.contourArea(contornos_0[cont])

        # filtro por children
        contornos_dibujados = cv.drawContours(
            frame, contornos_0, -1, (255, 0, 0), 1)
        cv.imshow('image', contornos_dibujados)
        cv.waitKey(1)
        indx_3 = np.where((CI_area_array[:]) < 50)
        contornos_filtrado_smallArea = np.take(contornos_0, indx_3)

        # contornos_filtrado_area = np.take(contornos_0, indx_2)

        contornos_dibujados_2 = cv.drawContours(
            frame, contornos_filtrado_smallArea[0, :], -1, (255, 0, 0), 2)

        cv.imshow('image', contornos_dibujados_2)
        cv.waitKey(1)

        for pixels in range(len(contornos_filtrado_smallArea)):
            for pixel in contornos_filtrado_smallArea[pixels]:
                for p in pixel:
                    threshold_image[p[0, 1], p[0, 0]] = 0

        cv.imshow('image', threshold_image)
        cv.waitKey(1)

        #########################################################################

        #########################################################################
        contornos_final, hierarchy = cv.findContours(
            threshold_image, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

        contornos_dibujados_2 = cv.drawContours(
            frame, contornos_final, -1, (0, 0, 255), 1)
        cv.imshow('image', contornos_dibujados_2)

        CI_area_array_final = np.ndarray(
            (len(contornos_final)), dtype=np.int32)
        # filtro por area
        for cont in range(len(contornos_final)):
            CI_area_array_final[cont] = cv.contourArea(contornos_final[cont])

        indx = np.where((hierarchy[0, :, 3]) == -1)
        indx = np.intersect1d(indx, indx)

        CI_area_array_final_filtred = np.take(contornos_final, indx)
        contornos_dibujados_2 = cv.drawContours(
            frame, CI_area_array_final_filtred, -1, (0, 255, 0), 1)
        cv.imshow('image', contornos_dibujados_2)

        CI_array = np.ndarray(
            (len(CI_area_array_final_filtred)), dtype=np.object0)

        for arrayCounter in range(len(CI_area_array_final_filtred)):
            ca = np.asarray(
                CI_area_array_final_filtred[arrayCounter], dtype=np.int16)
            CI_array[arrayCounter] = ca  # ca.swapaxes(1, 2)[:]

        cv.imwrite('Data/'+str(ncarpeta+1)+'/'+str(nvideo+1)+'/' +
                   'Total_Contour/'+str(f)+'.png', threshold_image)
        cv.imwrite('Data/'+str(ncarpeta+1)+'/'+str(nvideo+1)+'/' +
                   'Inner_Contour/'+str(f)+'.png', contornos_dibujados)

        CI_array_totalFrame[f] = CI_array

        # plt.hist(gray_image.ravel(), 256)

        # if cv.waitKey(1) == ord('q'):
        #   break

        cv.imshow('image', contornos_dibujados_2)

        cv.waitKey(1)
    return
# 33


def contourDetect_1(frame, cropArea, low_H, low_S, low_V, high_H, high_S, high_V, ncarpeta, nvideo, f, filter_min_area):

    frame = frame[cropArea[1]:cropArea[3], cropArea[0]:cropArea[2]]
    gray_image = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_HSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # threshold_image = backSub.apply(frame)

    ret2, th0 = cv.threshold(
        gray_image, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

    blur = cv.bilateralFilter(gray_image, 10, 10, 10)
    # Otsu's thresholding after Gaussian filtering
    ret2, th1 = cv.threshold(
        blur, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

    th2 = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_MEAN_C,
                               cv.THRESH_BINARY_INV, 11, 2)

    th3 = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv.THRESH_BINARY_INV, 11, 2)

    # threshold_image_1 = cv.bitwise_not(threshold_image_1)
    threshold_image_1 = cv.bitwise_or(th2, th3)
    threshold_image_2 = cv.bitwise_or(th0, th1)

    # SImple thresholding
    # 122 es el valor arbitrario
    # ret, th001 = cv.threshold(gray_image, 122, 255, cv.THRESH_BINARY_INV)

    # esta imagen binaria es calculada mediante metodos adaptativos
    threshold_image_n = cv.bitwise_or(threshold_image_1, threshold_image_2)

    frame_threshold_HSV = cv.inRange(
        frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))
    frame_threshold_HSV = cv.bitwise_not(frame_threshold_HSV)

    threshold_image = cv.bitwise_or(threshold_image_n, frame_threshold_HSV)

    # CI_Mask = cv.bitwise_not(threshold_image)

    #  contornos
    # contours = np.zeros((1, 1), dtype=np.object0)

    contornos_0, hierarchy = cv.findContours(
        threshold_image, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    CI_area_array = np.ndarray(
        (len(contornos_0)), dtype=np.int32)
    # filtro por area
    for cont in range(len(contornos_0)):
        CI_area_array[cont] = cv.contourArea(contornos_0[cont])

    indx_3 = np.where((CI_area_array[:]) < filter_min_area)
    contornos_filtrado_smallArea = np.take(contornos_0, indx_3)

    for pixels in range(len(contornos_filtrado_smallArea)):
        for pixel in contornos_filtrado_smallArea[pixels]:
            for p in pixel:
                threshold_image[p[0, 1], p[0, 0]] = 0

    #########################################################################

    #########################################################################
    contornos_final, hierarchy = cv.findContours(
        threshold_image, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    CI_area_array_final = np.ndarray(
        (len(contornos_final)), dtype=np.int32)
    # filtro por area
    for cont in range(len(contornos_final)):
        CI_area_array_final[cont] = cv.contourArea(contornos_final[cont])

    indx_1 = np.argwhere((hierarchy[0, :, 3]) == -1)
    indx = indx_1.reshape([len(indx_1), ])
    CI_area_array_final_filtred = np.take(contornos_final, indx)

    # pasarlo a imagen

    threshold_image_filterparent = cv.drawContours(
        np.zeros([np.abs(cropArea[1]-cropArea[3]), np.abs(cropArea[0]-cropArea[2])], dtype=np.float32), CI_area_array_final_filtred, -1, [255], 1)

    th1 = np.uint8(threshold_image_filterparent)

    contornos_final, hierarchy = cv.findContours(
        th1, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    for cont in range(len(contornos_final)):
        CI_area_array_final[cont] = cv.contourArea(contornos_final[cont])

    indx_1 = np.argwhere((hierarchy[0, :, 2]) == -1)
    indx = indx_1.reshape([len(indx_1), ])
    CI_area_array_final_filtred = np.take(contornos_final, indx)

###################################################################################

    contornos_dibujados_2 = cv.drawContours(
        np.zeros([np.abs(cropArea[1]-cropArea[3]), np.abs(cropArea[0]-cropArea[2])], dtype=np.int8), CI_area_array_final_filtred, -1, 255, 1)

    CI_array = np.ndarray(
        (len(CI_area_array_final_filtred)), dtype=np.object0)

    for arrayCounter in range(len(CI_area_array_final_filtred)):
        ca = np.asarray(
            CI_area_array_final_filtred[arrayCounter], dtype=np.int16)
        CI_array[arrayCounter] = ca  # ca.swapaxes(1, 2)[:]

    cv.imwrite('Data/'+str(ncarpeta+1)+'/'+str(nvideo+1)+'/' +
               'Total_Contour/'+str(f)+'.png', threshold_image)
    cv.imwrite('Data/'+str(ncarpeta+1)+'/'+str(nvideo+1)+'/' +
               'Inner_Contour/' +
               str(f)+'.png', contornos_dibujados_2)

    return (CI_array)
###
###################################################


def contourDetect(frame, cropArea, low_H, low_S, low_V, high_H, high_S, high_V, ncarpeta, nvideo, f, filter_min_area):

    frame = frame[cropArea[1]:cropArea[3], cropArea[0]:cropArea[2]]
    gray_image = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_HSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # threshold_image = backSub.apply(frame)
    ret2, th0 = cv.threshold(
        gray_image, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    blur = cv.bilateralFilter(gray_image, 10, 10, 10)
    # Otsu's thresholding after Gaussian filtering
    ret2, th1 = cv.threshold(
        blur, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

    th2 = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_MEAN_C,
                               cv.THRESH_BINARY_INV, 11, 2)

    th3 = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv.THRESH_BINARY_INV, 11, 2)
    # threshold_image_1 = cv.bitwise_not(threshold_image_1)
    threshold_image_1 = cv.bitwise_or(th2, th3)
    threshold_image_2 = cv.bitwise_or(th0, th1)

    # SImple thresholding
    # 122 es el valor arbitrario
    # ret, th001 = cv.threshold(gray_image, 122, 255, cv.THRESH_BINARY_INV)

    # esta imagen binaria es calculada mediante metodos adaptativos
    threshold_image_n = cv.bitwise_or(threshold_image_1, threshold_image_2)

    frame_threshold_HSV = cv.inRange(
        frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))
    frame_threshold_HSV = cv.bitwise_not(frame_threshold_HSV)
    threshold_image = cv.bitwise_or(threshold_image_n, frame_threshold_HSV)


###################################################################################
#########Contornos################################Contornos########################
###################################################################################
#######Contornos##################Contornos########################Contornos#######
###################################################################################

###########################################
# Filtro de area
#    contornos_0, hierarchy = cv.findContours(
#        threshold_image, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
#    CI_area_array = np.ndarray(
#        (len(contornos_0)), dtype=np.int32)
#    for cont in range(len(contornos_0)):
#
#        CI_area_array[cont] = cv.contourArea(contornos_0[cont])
#
#    indx_area = np.where((CI_area_array[:]) < filter_min_area)
#    contornos_filtrado_smallArea = np.take(contornos_0, indx_area)
#    for pixels in range(len(contornos_filtrado_smallArea)):
#        for pixel in contornos_filtrado_smallArea[pixels]:
#            for p in pixel:
#                threshold_image[p[0, 1], p[0, 0]] = 0


# Filtro de area
###########################################
###########################################
# Filtro de padres

    contorno_padres, hierarchy = cv.findContours(
        threshold_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    #contorno_padres = np.asarray(contorno_padres)


# Filtro de padres
###########################################
###########################################
# Filtro de hijos

    # Primero obtenemos la imagen de los contornos padres
    threshold_image_padres = cv.drawContours(
        np.zeros([np.abs(cropArea[1]-cropArea[3]), np.abs(cropArea[0]-cropArea[2])], dtype=np.float32), contorno_padres, -1, [255], 1)

    # luego obtenemos la imagen del contorno total (padre mas hijo)

    contorno_total, hierarchy = cv.findContours(
        threshold_image, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    # la pasamos a una imagen
    threshold_image_total = cv.drawContours(
        np.zeros([np.abs(cropArea[1]-cropArea[3]), np.abs(cropArea[0]-cropArea[2])], dtype=np.float32), contorno_total, -1, [255], 1)

    # Ahora es posible restar los contornos totales y los contornos padre
    threshold_image_contorno_interno = np.uint8(
        threshold_image_total-np.uint8(threshold_image_padres))

    # para calcular el : contorno_hijos
    contorno_hijos, hierarchy = cv.findContours(
        threshold_image_contorno_interno, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    threshold_image_hijos = cv.drawContours(
        np.zeros([np.abs(cropArea[1]-cropArea[3]), np.abs(cropArea[0]-cropArea[2])], dtype=np.float32), contorno_hijos, -1, [255], 1)

    #contorno_hijos = np.asarray(contorno_hijos)

    # calculamos y filtramos el area == 0

    areaContorno_interno = list(
        map(lambda sub_lista: cv.contourArea(sub_lista), contorno_hijos))
    ind_bool = list(map(lambda sub_lista: sub_lista > 0, areaContorno_interno))
    ind = [areaContorno_interno for areaContorno_interno,
           x in enumerate(ind_bool) if x]
    AF = [areaContorno_interno[i] for i in ind]

    CI = [contorno_hijos[i] for i in ind]

    M = list(map(lambda sub_lista: cv.moments(sub_lista), CI))

    M00 = np.asarray(list(map(lambda sub_lista: sub_lista['m00'], M)))
    M10 = np.asarray(list(map(lambda sub_lista: sub_lista['m10'], M)))
    M01 = np.asarray(list(map(lambda sub_lista: sub_lista['m01'], M)))

    Cx = M10/M00
    Cy = M01/M00

    BR_ = list(map(lambda sub_lista: cv.boundingRect(sub_lista), contorno_hijos))
    BR = [BR_[i] for i in ind]
    #Bounding_Rect == x, y, w, h

# np.ndarray(len(areas), dtype=np.object0)
# for ind in range(len(areas)):
#   np.argwhere(Contornos == ind)

# Para obtener los indices

# cv.imshow('threshold_image_hijos', threshold_image)
# basicamente restar threshold image - contorno_padres_final


# Filtro de hijos
###########################################
###########################################


###################################################################################
# Contornos########################Contorn
###################################################################################
# Contornos########################Contornos###########Con
###################################################################################
###################################################################################

    cv.imwrite('Data/'+str(ncarpeta+1)+'/'+str(nvideo+1)+'/' +
               'Total_Contour/'+str(f)+'.png', threshold_image_padres)
    cv.imwrite('Data/'+str(ncarpeta+1)+'/'+str(nvideo+1)+'/' +
               'Inner_Contour/' +
               str(f)+'.png', threshold_image_contorno_interno)
    
    
    CI = np.array(CI,dtype=np.object_)
    contorno_padres= np.array(contorno_padres,dtype=np.object_)
    BR= np.array(BR,dtype=np.object_)
    AF= np.array(AF,dtype=np.int32)
    Cx= np.array(Cx,dtype=np.int32)
    Cy= np.array(Cy,dtype=np.int32)
    return (CI, contorno_padres, BR, AF, Cx, Cy)


""" 
    Propiedades_CI = cv.connectedComponentsWithStats(Contorno_interno, 8)

    XX, YY = np.meshgrid(np.arange(Contorno_interno.shape[1]), np.arange(
    
    
    Contorno_interno.shape[0]), copy=False)
    Contorno_interno_RAVEL = np.ravel(Contorno_interno)
    Contorno_interno_RAVEL=np.asarray(Contorno_interno_RAVEL)
    YRAVEL = np.ravel(YY)
    XRAVEL = np.ravel(XX)
    nonzero = np.where(Contorno_interno_RAVEL > 0)

    borrow = np.concatenate(([0], (nonzero[0][0:len(nonzero[0])-1])))
    resta = nonzero[0]-borrow

    index_split = np.where(resta > 1)
    
    xd=np.split(Contorno_interno_RAVEL,nonzero[0][index_split[0][1:]])
    
    Contornos_Externos = np.split(Contorno_interno_RAVEL, index_split[0][1:])

    Contorno_externo_flat = np.nonzero(Contorno_externo.flatten()) 

areas = np.asarray(Propiedades_CI[2][:, 4], dtype=np.int32)
# maximo numero de area logica permitida
ind = np.where(areas < 30000)
ind_Burbuja = ind[0]

centroides = np.asarray(Propiedades_CI[3], dtype=np.float32)

CI = np.asarray(Propiedades_CI[1], dtype=np.int32)

"""


def area_contour(contour):
    contour = cv.contourArea(contour)
    return contour


def Video_Threshold(video_folder, cropArea, ncarpeta, nvideo, low_H, high_H, low_S, high_S, low_V, high_V):

    video = cv.VideoCapture(video_folder)

    TOTAL_FRAME = video.get(cv.CAP_PROP_FRAME_COUNT)
    TOTAL_FRAME = int(TOTAL_FRAME)
    filter_min_area = 50

    CI_array_totalFrame = np.ndarray(TOTAL_FRAME, dtype=np.object_)
    CE_array_totalFrame = np.ndarray(TOTAL_FRAME, dtype=np.object_)

    BR_array_totalFrame = np.ndarray(TOTAL_FRAME, dtype=np.object_)
    AF_array_totalFrame = np.ndarray(TOTAL_FRAME, dtype=np.object_)
    Cx_array_totalFrame = np.ndarray(TOTAL_FRAME, dtype=np.object_)
    Cy_array_totalFrame = np.ndarray(TOTAL_FRAME, dtype=np.object_)

    for f in range(100):  # range(TOTAL_FRAME):

        ret, frame = video.read()

        (CI, CE, BR, AF, Cx, Cy) = contourDetect(
            frame, cropArea, low_H, low_S, low_V, high_H, high_S, high_V, ncarpeta, nvideo, f, filter_min_area)
        CI_array_totalFrame[f] = CI
        BR_array_totalFrame[f] = BR
        AF_array_totalFrame[f] = AF
        Cx_array_totalFrame[f] = Cx
        Cy_array_totalFrame[f] = Cy
        CE_array_totalFrame[f] = CE

    return (CI_array_totalFrame,BR_array_totalFrame,AF_array_totalFrame)
                                                                                                            
CI_array_totalFrame, BR_array_totalFrame, AF_array_totalFrame = Video_Threshold('1/1.MP4', np.array([275,  207, 1562,  840]), 0, 0, 0, 180, 0, 255, 138, 255)
# Video_Threshold_Debug('1/1.MP4', np.array([[[409], [332], [1444], [769]]]), 0, 0)

np.save('CI_array_totalFrame',CI_array_totalFrame)
np.save('BR_array_totalFrame',BR_array_totalFrame)
np.save('AF_array_totalFrame',AF_array_totalFrame)
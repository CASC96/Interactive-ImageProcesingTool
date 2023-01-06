import cv2

x_start, y_start, x_end, y_end = 0, 0, 0, 0
cropping = False


def mouse_crop(event, x, y, flags, param):
    # grab references to the global variables
    global x_start, y_start, x_end, y_end, cropping
    # if the left mouse button was DOWN, start RECORDING
    # (x, y) coordinates and indicate that cropping is being
    if event == cv2.EVENT_LBUTTONDOWN:
        x_start, y_start, x_end, y_end = x, y, x, y
        cropping = True

    # Mouse is Moving
    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping == True:
            x_end, y_end = x, y

    # if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates
        x_end, y_end = x, y
        cropping = False  # cropping is finished


def cropcs(image):

    #image = cv2.imread(image)
    cv2.namedWindow("image", cv2.WINDOW_GUI_NORMAL)
    cv2.setWindowProperty(
        "image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("image", image)
    while True:
        i = image.copy()
        cv2.setMouseCallback("image", mouse_crop)
        if not cropping:
            pass
        elif cropping:
            cv2.rectangle(i, (x_start, y_start),
                          (x_end, y_end), (255, 0, 0), 2)
            cv2.imshow("image", i)
        k = cv2.waitKey(1)
        if k == ord('q'):

            break
    cv2.waitKey(0)
    # close all open windows
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    return(x_start, y_start, x_end, y_end)

import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt


def perform_canny(image,min_th,max_th):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, min_th, max_th)
    return canny

def regions_of_interest(image):
    height = image.shape[0]
    polygons = np.array([
    [(200,height),(1100,height),(550,250)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask,polygons,255)
    masked_image = cv2.bitwise_and(mask,image)
    return masked_image

def detect_lines(image):
    lines = cv2.HoughLinesP(image,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=1)
    return lines

def make_coordinates(y1,y2,line_params):
    slope = line_params[0]
    intercept = line_params[1]
    x1 = int((y1-intercept)/slope)
    x2 = int((y2-intercept)/slope)
    return np.array([x1,y1,x2,y2])


def calc_average_lines(image,lines):
    right_side_lines = []
    center_side_lines = []


    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2),(y1,y2),1)
        slope = parameters[0]
        intercept = parameters[1]

        if slope<0:
            center_side_lines.append((slope,intercept))

        else:
            right_side_lines.append((slope,intercept))

    right_params_average = np.average(right_side_lines, axis=0)
    center_params_average = np.average(center_side_lines, axis=0)


    center_y1 = image.shape[0]
    center_y2 = int(center_y1*(3/5))
    right_y1 = image.shape[0]
    right_y2 = int(right_y1*(3/5))
    right_line = right_line = make_coordinates(right_y1,right_y2,right_params_average)
    try:
        center_line = make_coordinates(center_y1,center_y2,center_params_average)
    except:
        return np.array([right_line])

    return np.array([center_line,right_line])


def display(image,lines):
    lines_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)
            cv2.line(lines_image,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),10)

    return lines_image



path = sys.argv[1]
original_video = cv2.VideoCapture(path)

width = original_video.get(cv2.CAP_PROP_FRAME_WIDTH )
height = original_video.get(cv2.CAP_PROP_FRAME_HEIGHT )
fps =  original_video.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video = cv2.VideoWriter(filename ='output.avi', fourcc=fourcc ,fps=int(fps), frameSize=(int(width),int(height)),apiPreference=None,params=None)


while original_video.isOpened():

    _,frame = original_video.read()
    canny = perform_canny(frame,50,150)
    masked_canny = regions_of_interest(canny)
    lines = detect_lines(masked_canny)
    average_lines = calc_average_lines(frame,lines)
    lines_image = display(frame,average_lines)

    final_image = cv2.addWeighted(frame,0.8,lines_image,1,1)


    cv2.imshow('Final',final_image)

    output_video.write(final_image)

    if cv2.waitKey(1) == ord('q'):
        break

original_video.release()
output_video.release()
cv2.destroyAllWindows()


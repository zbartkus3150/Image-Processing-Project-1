import numpy as np
import cv2
import glob
import os
import random
from math import sqrt, floor


#Function for creating the mask
#if debug is true then the output image of each step is shown and saved
def make_mask(path, debug):

    #read image
    image = cv2.imread(path,cv2.IMREAD_COLOR)

    if debug: 
        cv2.imshow("Image", image)
        save(image, 'Image.png', './example')

    #Remove every color which is not in range
    greenMin = np.array([10 ,20 ,10])
    greenMax = np.array([130 ,255 ,150])
    mask = cv2.inRange(image, greenMin, greenMax)

    if debug: 
        cv2.imshow("Mask", mask)
        save(mask, 'Mask_1.0.png', './example')
    #Apply first blur to eliminate small noise for contour finding
    mask = cv2.medianBlur(mask , 3)

    if debug: 
        cv2.imshow("Mask_blur", mask)
        save(mask, 'Mask_1.1.png', './example')

    #Find contours
    ret,thresh = cv2.threshold(mask, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #Sort contours and find the closest to the center of the image using moments
    plant_cont = contours[0]
    min=image.shape[0]
    for i in range(len(contours)):
        if cv2.contourArea(contours[i]) > 500: 
            M = cv2.moments(contours[i])
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            Z = sqrt((cX - image.shape[1]/2)**2 + (cY - image.shape[0]/2)**2)
            if min > Z: 
                min=Z
                plant_cont=contours[i]
    
    #Compute the rotated bounding box of the found contour
    rect = cv2.minAreaRect(plant_cont)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    #Draw bounding box
    image_box = image.copy()
    cv2.drawContours(image_box, [box], -1, (0, 255, 0), 2)

    if debug: 
        cv2.imshow("box", image_box)
        save(image_box, 'Box.png', './example')
    
    #Draw contours and their insides on the empty image to obtain only plant on the mask 
    mask = cv2.drawContours(np.zeros((image.shape[0], image.shape[1], 3), np.uint8), [plant_cont], 0, (255,255,255), cv2.FILLED)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    #apply bitwise AND oparation to get segmented plant
    segmented = cv2.bitwise_and(image , image , mask=mask) 
    
    if debug: 
        cv2.imshow("seg", segmented)
        save(segmented, 'Segmented.png', './example')
        save(mask, 'One_object.png', './example')

    #Use narrower range to eliminate non-plant objects like pot and shadows
    hsv = cv2.cvtColor(segmented , cv2.COLOR_BGR2HSV) 
    potMin = np.array([35, 30, 30],np.uint8) 
    potMax = np.array([90, 255, 135],np.uint8)
    mask = cv2.inRange(hsv, potMin , potMax)


    #Apply morphology such as opening and closure
    kernel = np.ones((1,1),np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN , kernel)
    mask = cv2.morphologyEx(mask ,cv2.MORPH_CLOSE ,kernel)

    if debug: 
        cv2.imshow("Mask Segmented", mask)
        save(mask, 'Mask_2.0.png', './example')

    #Apply second blur and series of erosion and dilation depending on plant area
    iter = 1
    if cv2.contourArea(plant_cont) > 30000:
        iter = 3
    mask = cv2.medianBlur(mask , 3)
    if debug: 
        cv2.imshow("Mask Segmented 2", mask)
        save(mask, 'Mask_3.0.png', './example')

    mask = cv2.erode(mask, None, iterations = int(iter))
    mask = cv2.dilate(mask, None, iterations = int(iter))

    if debug: 
        cv2.imshow("Erosion - Dilation", mask)
        save(mask, 'Mask_4.0.png', './example')
        cv2.waitKey(0)

    return mask, image_box







#Function for painting leaves in mask on different colors
#if debug is true then the output image of each step is shown and saved
def label_mask(path, name, debug):
    #read mask
    label = cv2.imread(path + '/' + name ,cv2.IMREAD_GRAYSCALE)
    label_dilate = cv2.dilate(label, None, iterations=2)
    ret,label_thresh = cv2.threshold(label_dilate, 127, 255, 0)

    #Get coordinates of all white pixels and turn them into list of tuples 
    indices = np.where(label==[255])
    coords = zip(indices[1], indices[0])
    coords = list(coords)

    
    #Find contours and get the largest one
    contours_label, hierarchy = cv2.findContours(label_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_label = sorted(contours_label, key = cv2.contourArea, reverse = True)[0]

    #Compute the center of the biggest object using moments
    M_label = cv2.moments(max_label)
    cX_label = int(M_label["m10"] / M_label["m00"])
    cY_label = int(M_label["m01"] / M_label["m00"])

    #Adjust variables depending on the size of object
    area = cv2.contourArea(max_label)
    if area > 1000 and area < 4000:
        radius = 10
        scale = floor(area/2000)
    else:
       if area < 10000:
           radius = 20
           scale = floor(area/5000)
       else:
           radius = 30
           scale = floor(area/8000)
    if area > 40000:
        scale = floor(scale/2)

    #Draw black circle for proportional size on the center of object to eliminate pot and stalks
    cv2.circle(label, (cX_label,cY_label), radius, color=(0,0,0), thickness=-1)

    if debug: 
        cv2.imshow("circle", label)
        save(label, 'circle.png', './example2')
    

    #Apply series of erosions and openings to get rid of remaining stalks and obtain separate leaves
    kernel = np.ones((7+scale,7+scale),np.uint8)
    eroded = cv2.erode(label, kernel, iterations = 1)
    eroded = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, kernel/2, iterations=1)
    eroded = cv2.medianBlur(eroded, 5)

    #Find contours of each element
    ret,thresh = cv2.threshold(eroded, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if debug: 
        cv2.imshow("erode", eroded)
        save(eroded, 'eroded.png', './example2')

    #Draw empty image
    zeros = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)

    #Our color palette
    colors=[(255,0,0),(0,255,0),(255,255,0),(0,0,255),(255,0,255),(0,255,255),(128,128,128),(128,0,0)]

    #Compute distance from each pixel to centres of leaves 
    cX = []
    cY = []
    for i in range(len(contours)):
        M = cv2.moments(contours[i])
        cX.append(int(M["m10"] / M["m00"]))
        cY.append(int(M["m01"] / M["m00"]))
    
    for xy in coords:
        min = label.shape[0]
        index = 0
        for i in range(len(contours)):
            #Pythagorean theorem for points on the plane
            dist = sqrt(((xy[0] - cX[i])**2) + ((xy[1] - cY[i])**2))
            if dist < min:
                min = dist
                index = i
        #"Paint" each pixel on specific colour depending on the distance by drawing a circle with 1-pixel radius on it
        cv2.circle(zeros, xy, 1, color=colors[index])
        zeros[xy[0]][xy[1]] = colors[index]
        if(xy[0] >= label.shape[0] and xy[1] >= label.shape[1]):
            zeros[label.shape[0]-1][label.shape[1]-1] = colors[index]
        else:
            if xy[0] >= label.shape[0]:
                zeros[label.shape[0]-1][xy[1]] = colors[index]
            else:
                if xy[1] >= label.shape[1]:
                    zeros[xy[0]][label.shape[1]-1] = colors[index]
                else:
                    zeros[xy[0]][xy[1]] = colors[index]
    
    if debug:
        cv2.imshow("Zeros", zeros)
        save(zeros, 'label.png', './example2')
        cv2.waitKey()

    return zeros






#Function for saving the image
def save(image, name, path):
    name = os.path.basename(name)
    cv2.imwrite(path + '/' + name, image)






#Function for reading image and turning it into mask 
def read(path, name):
    label = cv2.imread(path + '/' + name)
    label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
    ret, label = cv2.threshold(label, 0, 255, cv2.THRESH_BINARY)
    return label





#Function for reading image in color and turning each color into separate mask
def readColor(path, name):
    colors=[(255,0,0),(0,255,0),(255,255,0),(0,0,255),(255,0,255),(0,255,255),(128,128,128),(128,0,0)]
    label = cv2.imread(path + '/' + name)
    masks = []
    for color in colors:
        c = np.array(color)
        mask = cv2.inRange(label, c, c)
        masks.append(mask)
    return masks





#Function for creating and saving masks, bounding boxes and labels for each plant image
def make_All():
    readPathPlant = './multi_plant'
    writePathMask = './masks'
    writePathBox = './boxes'
    writePathLabel = './labels'

    files = [f for f in glob.glob(readPathPlant + "**/*.png", recursive=True)] 
    for f in files:
        mask, box = make_mask(f, False)
        save(mask, f, writePathMask)
        save(box, f.replace('rgb', 'box'), writePathBox)
        name = os.path.basename(f)
        mylabel = label_mask(writePathMask, name, False)
        save(mylabel, f.replace('rgb', 'label'), writePathLabel)
        print(name + " saved!")
    
    print("All files saved")
    cv2.waitKey()





#Function for comparing our masks with masks from labels
def compare_masks():
    print("---------------------------Comparison of masks---------------------------\n")

    readPathLabel = './multi_label'
    writePathMask = './masks'

    files_mask = [f for f in glob.glob(writePathMask + "**/*.png", recursive=True)] 
    mean_IoU = 0
    mean_Dice = 0
    min_IoU = 1
    max_IoU = 0
    min_Dice = 1
    max_Dice = 0
    results = []
    
    for f in files_mask:
        name = os.path.basename(f)
        mask = read(writePathMask, name)
        label = read(readPathLabel, name.replace('rgb', 'label'))

        Dice = np.sum(mask[label==255])*2.0 / (np.sum(mask) + np.sum(label))
        IoU = np.sum(cv2.bitwise_and(mask, label)) / np.sum(cv2.bitwise_or(mask, label))

        results.append((name, IoU, Dice))
        print(name + ' :  IoU: ' + str(IoU) + '   Dice: ' + str(Dice))
        mean_IoU += IoU
        mean_Dice += Dice
        if IoU > max_IoU:
            max_IoU = IoU
            nameMaxIoU = os.path.basename(f)
        if IoU < min_IoU:
            min_IoU = IoU
            nameMinIoU = os.path.basename(f)
        if Dice > max_Dice:
            max_Dice = Dice
            nameMaxDice = os.path.basename(f)
        if Dice < min_Dice:
            min_Dice = Dice
            nameMinDice = os.path.basename(f)

    mean_IoU /= 900
    mean_Dice /= 900
    print("\n-------------------------------Jaccard index------------------------------\n")
    print("Mean IoU: " + str(mean_IoU))
    print("Min IoU: " + nameMinIoU + ' - ' + str(min_IoU))
    print("Max IoU: " + nameMaxIoU + ' - ' + str(max_IoU))
    print("\n------------------------------Dice coefficient----------------------------\n")
    print("Mean Dice: " + str(mean_Dice))
    print("Min Dice: " + nameMinDice + ' - ' + str(min_Dice))
    print("Max Dice: " + nameMaxDice + ' - ' + str(max_Dice) + "\n")
    with open('results_masks.txt', 'w') as f:
        f.write("------------------------Comparison of masks---------------------\n\n")
        f.write("	Filename: 	      IoU score:         Dice Score:\n")
        for item in results:          
            f.write("%s\n" % str(item))
        f.write("\n%s\n\n" % "----------------------------Jaccard index-----------------------")
        f.write("%s" % "Mean IoU: " + str(mean_IoU) + "\n")
        f.write("%s" % "Min IoU: " + nameMinIoU + ' - ' + str(min_IoU) + "\n")
        f.write("%s" % "Max IoU: " + nameMaxIoU + ' - ' + str(max_IoU) + "\n")
        f.write("\n%s\n\n" % "---------------------------Dice coefficient---------------------")
        f.write("%s" % "Mean Dice: " + str(mean_Dice) + "\n")
        f.write("%s" % "Min Dice: " + nameMinDice + ' - ' + str(min_Dice) + "\n")
        f.write("%s" % "Max Dice: " + nameMaxDice + ' - ' + str(max_Dice) + "\n")

    cv2.waitKey()




#Function for comapring our prediction labels with sample labels
def compare_labels():
    print("----------------------------Comparison of labels-----------------------------")

    readPathLabel = './multi_label'
    writePathLabel = './labels'

    files_labels = [f for f in glob.glob(readPathLabel + "**/*.png", recursive=True)] 
    mean_IoU = 0
    mean_Dice = 0
    min_IoU = 1
    max_IoU = 0
    min_Dice = 1
    max_Dice = 0
    results = []

    
    for f in files_labels:
        name = os.path.basename(f)
        my_label = readColor(writePathLabel, name)
        label = readColor(readPathLabel, name)
        IoU = []
        Dice = []
        leaves = 0
        for i in range(8):
            if(np.sum(cv2.bitwise_and(my_label[i], label[i])) != 0):
                IoU.append(np.sum(cv2.bitwise_and(my_label[i], label[i])) / np.sum(cv2.bitwise_or(my_label[i], label[i])))
            else:
                IoU.append(0)
                leaves = leaves + 1

            if np.sum(my_label[i][label[i]==255]) != 0:
                Dice.append(np.sum(my_label[i][label[i]==255])*2.0 / (np.sum(my_label[i]) + np.sum(label[i])))
            else:
                Dice.append(0)

        if leaves == 8:
            leaves = 7
        IoU_sum = sum(IoU) / (8-leaves)
        Dice_sum = sum(Dice) / (8-leaves)
        results.append((name, IoU, IoU_sum, Dice, Dice_sum))
        print(name + ' :  IoU: ' + str(IoU_sum) + '   Dice: ' + str(Dice_sum))
        mean_IoU += IoU_sum
        mean_Dice += Dice_sum
        if IoU_sum > max_IoU:
            max_IoU = IoU_sum
            nameMaxIoU = os.path.basename(f)
        if IoU_sum < min_IoU:
            min_IoU = IoU_sum
            nameMinIoU = os.path.basename(f)
        if Dice_sum > max_Dice:
            max_Dice = Dice_sum
            nameMaxDice = os.path.basename(f)
        if Dice_sum < min_Dice:
            min_Dice = Dice_sum
            nameMinDice = os.path.basename(f)

    mean_IoU /= 900
    mean_Dice /= 900
    print("\n----------------------Jaccard index------------------------\n")
    print("Mean IoU: " + str(mean_IoU))
    print("Min IoU: " + nameMinIoU + ' - ' + str(min_IoU))
    print("Max IoU: " + nameMaxIoU + ' - ' + str(max_IoU))
    print("\n---------------------Dice coefficient----------------------\n")
    print("Mean Dice: " + str(mean_Dice))
    print("Min Dice: " + nameMinDice + ' - ' + str(min_Dice))
    print("Max Dice: " + nameMaxDice + ' - ' + str(max_Dice) + "\n")
    with open('results_labels.txt', 'w') as f:
        f.write("--------------------------------Comparison of labels---------------------------------\n")
        f.write("	Filename: 	      IoU per leaf score:         IoU score:			Dice per leaf score:				Dice score:\n")
        for item in results:
            f.write("%s\n" % str(item))
        f.write("\n%s\n\n" % "--------------------------Jaccard index-------------------------------------------")
        f.write("%s" % "Mean IoU: " + str(mean_IoU) + "\n")
        f.write("%s" % "Min IoU: " + nameMinIoU + ' - ' + str(min_IoU) + "\n")
        f.write("%s" % "Max IoU: " + nameMaxIoU + ' - ' + str(max_IoU) + "\n")
        f.write("\n%s\n\n" % "------------------------Dice coefficient------------------------------------------")
        f.write("%s" % "Mean Dice: " + str(mean_Dice) + "\n")
        f.write("%s" % "Min Dice: " + nameMinDice + ' - ' + str(min_Dice) + "\n")
        f.write("%s" % "Max Dice: " + nameMaxDice + ' - ' + str(max_Dice) + "\n")
    cv2.waitKey()



def main():
    print("Computer Vision project 1")

    #Tests:
    #make_mask(path, True)
    #label_mask('./masks', 'rgb_00_00_000_05.png', True)
    #label_mask('./masks', 'rgb_01_02_001_00.png', True)

    #Main code:
    make_All()
    compare_masks()
    compare_labels()

    print("The End")


main()



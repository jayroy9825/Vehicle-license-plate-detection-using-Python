


import cv2
import numpy as np
import os

import DetectChars
import DetectPlates
import PossiblePlate

from PIL import Image
import pytesseract
import argparse
from tkinter import *
import tkinter.messagebox
from PIL import Image, ImageTk
import csv
pytesseract= r'/usr/local/Cellar/tesseract/4.1.1/bin/tesseract'

#colour
SCALAR_BLACK=(0,0,0)
SCALAR_WHITE=(255,255,255)
SCALAR_YELLOW=(0,255,255)
SCALAR_GREEN=(0,255,0)
SCALAR_TEAL=(72,100,50)
SCALAR_RED=(255,0,0)


def hasNumbers(mystring):
    return any(char.isdigit() for char in mystring)
    

def main():

    blnKNNTrainingSuccessful = DetectChars.loadKNNDataAndTrainKNN()

    if blnKNNTrainingSuccessful == False:
        print("\nerror: KNN traning was not successful\n")
        return
    path1=path='dataset/images/90090.jpg'
    path1=(path1.split('/'))
    k=((path1[(len(path1)-1)]).split('.')[0])+".png"
    imgOriginalScene  = cv2.imread(path)

    if imgOriginalScene is None:
        print("\nerror: image not read from file \n\n")
        os.system("pause")
        return
    else:

        listOfPossiblePlates = DetectPlates.detectPlatesInScene(imgOriginalScene)

        listOfPossiblePlates = DetectChars.detectCharsInPlates(listOfPossiblePlates)

        cv2.imshow("imgOriginalScene", imgOriginalScene)
        cv2.waitKey(0)
        flag=0
        
        with open("annot_file1.csv","r") as f2:
            f1 = csv.reader(f2, delimiter=',')
            l=[]
            for row in f1:
                #print(row)
                l.append(row)

            for i in range(0,len(l)):
                if l[i][2]==k:
                    text=l[i][3]
                    xmin=l[i][4]
                    ymin=l[i][5]
                    xmax=l[i][6]
                    ymax=l[i][7]
                    flag=1
                    break
    
            if (flag==0):
                if len(listOfPossiblePlates) == 0:
                    print("\nno license plates were detected\n")
                else:
                    listOfPossiblePlates.sort(key = lambda possiblePlate: len(possiblePlate.strChars), reverse = True)
                    licPlate = listOfPossiblePlates[0]
            
                cv2.imwrite('dataset/crops/'+k,licPlate.imgPlate)
                cv2.waitKey(0)
                gray = cv2.cvtColor(licPlate.imgPlate, cv2.COLOR_BGR2GRAY)
                gray = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
                
                if len(licPlate.strChars) == 0:
                    print("\nno characters were detected\n\n")
                    return
                xmin,ymin,xmax,ymax=drawRedRectangleAroundPlate(imgOriginalScene,licPlate)
                
                text=licPlate.strChars
                
                with open("annot_file1.csv",'a') as f3:
                    writer = csv.writer(f3,delimiter = ',')
                    writer.writerow([(int((l[len(l)-1])[0])+1),(int((l[len(l)-1])[1])+1),k,licPlate.strChars,int(xmin),int(ymin),int(xmax),int(ymax)])
                # cv2.imwrite("imgOriginalScene.png", imgOriginalScene)
            if flag==1:
                imgOriginalScene= cv2.rectangle(imgOriginalScene,(int(xmin),int(ymax)),(int(xmax),int(ymin)),(255, 0, 0),2) 
            
            print("----------------------------------------")
            print("\nlicense plate number = " +text+ "\n")
            print("----------------------------------------")
            test=cv2.imread('dataset/crops/'+k)
            cv2.imshow("test",test)
            writeLicensePlateCharsOnImage1(imgOriginalScene,xmin,ymin,text,test)
            cv2.imshow("imgOriginalScene", imgOriginalScene)
            

    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    k = cv2.waitKey(0) & 0xFF
  
    # wait for ESC key to exit 
    if k == 27:  
        cv2.destroyAllWindows() 
    return



def drawRedRectangleAroundPlate(imgOriginalScene, licPlate):

    p2fRectPoints = cv2.boxPoints(licPlate.rrLocationOfPlateInScene)
    

    cv2.line(imgOriginalScene, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), SCALAR_RED, 2)
    return (list(p2fRectPoints[0]))[0],(list(p2fRectPoints[1]))[1],(list(p2fRectPoints[2]))[0],(list(p2fRectPoints[3]))[1]


def writeLicensePlateCharsOnImage1(imgOriginalScene,xmin,ymin,text,test):
    ceneHeight, sceneWidth, sceneNumChannels = imgOriginalScene.shape
    plateHeight, plateWidth, plateNumChannels = test.shape

    intFontFace = cv2.FONT_HERSHEY_SIMPLEX
    fltFontScale =1
    intFontThickness = int(round(fltFontScale * 1.5))
    
    #intFontFace = cv2.FONT_HERSHEY_SIMPLEX
    #fltFontScale = .8
    #intFontThickness = 2
    cv2.putText(imgOriginalScene, text, (int(xmin)-21,int(ymin)), intFontFace, fltFontScale,SCALAR_TEAL, intFontThickness)




if __name__ == "__main__":
    main()
'''if __name__ == '__main__':
    root = Tk()
    tool = LabelTool(root)
    root.resizable(width =  True, height = True)
    root.mainloop()'''
          







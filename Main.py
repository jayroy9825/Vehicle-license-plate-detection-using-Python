import cv2
import numpy as np
import os

import DetectChars
import DetectPlates
import PossiblePlate
import argparse
import tkinter.messagebox
from PIL import Image, ImageTk
import csv
from tkinter import *
from PIL import Image
from PIL import ImageTk
import tkinter.filedialog as tkFileDialog


import wget
from tkinter import filedialog      

plate=""
image=""
plate_text=""

#colour
SCALAR_BLACK=(0,0,0)
SCALAR_WHITE=(255,255,255)
SCALAR_YELLOW=(0,255,255)
SCALAR_GREEN=(0,255,0)
SCALAR_TEAL=(72,100,50)
SCALAR_RED=(255,0,0)


def hasNumbers(mystring):
    return any(char.isdigit() for char in mystring)
    

#def main():
def select_image():
    
    # grab a reference to the image panels
    global panelA, panelB,image1,plate,plate_text

    # open a file chooser dialog and allow the user to select an input
    # image
    path = tkFileDialog.askopenfilename()

    # ensure a file path was selected
    if len(path) > 0:
        # load the image from disk, convert it to grayscale, and detect
        # edges in it   
        blnKNNTrainingSuccessful=DetectChars.loadKNNDataAndTrainKNN();
        if blnKNNTrainingSuccessful == False:
            print("\nerror: KNN traning was not successful\n")
            return
        #path1=path='dataset/images/90090.jpg'
        path1=path
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
    
            #cv2.imshow("imgOriginalScene", image)
            #cv2.waitKey(0)
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
                    #cv2.waitKey(0)
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
                plate_text=text
                test=cv2.imread('dataset/crops/'+k)
               
                writeLicensePlateCharsOnImage1(imgOriginalScene,xmin,ymin,text,test)
               
                # convert the images to PIL format...
               
                plate = cv2.resize(test,(200,80))
               
                image1 = cv2.resize(imgOriginalScene,(390,240))
                # OpenCV represents images in BGR order; however PIL represents
                # images in RGB order, so we need to swap the channels
                plate = cv2.cvtColor(plate, cv2.COLOR_BGR2RGB)
                image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
                plate= Image.fromarray(plate)
                image1= Image.fromarray(image1)
        
                # ...and then to ImageTk format
                plate= ImageTk.PhotoImage(plate)
                image1= ImageTk.PhotoImage(image1)
                number(plate_text)
                # if the panels are None, initialize them
                if panelA is None or panelB is None:
                    
                    panelA= Label(MainFrame, image = image1)
                    panelA.grid(row=4,column=0,sticky=W)


                    panelB = Label(MainFrame, image =plate)
                    panelB.grid(row=4,column=1,sticky=W)
                    

        
                # otherwise, update the image panels
                else:
                    # update the pannels
                    panelA.configure(image=image1)
                    panelA.grid(row=4,column=0,sticky=W)
                    panelB.configure(image=plate)
                    panelB.grid(row=4,column=1,sticky=W)
                    panelA.image = image
                    panelA.grid(row=4,column=0,sticky=W)
                    panelB.image =plate
                    panelB.grid(row=4,column=1,sticky=W)
                
        
        



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
    cv2.putText(imgOriginalScene, text, (int(xmin)-21,int(ymin)), intFontFace, fltFontScale,SCALAR_WHITE, intFontThickness)


def number(num_detected):
    T = Text(MainFrame,font=('arial', 16,'bold'), height=5, width=20)
    s = "Extracted Number is :\n"
    T.insert(INSERT, (s+num_detected))
    T.grid(row=5,column=1,sticky=W)

root = Tk()
panelA = None
panelB = None

root.title("GUI : Number Plate")

root.geometry("880x530")

root.configure(background = 'white')
Tops = Frame(root,bg = 'blue',pady = 1, width =1850, height = 90, relief = "ridge")
Tops.grid(row=0,column=0)


Title_Label = Label(Tops,font=('Arial',20,'bold'),text = "         Number plate detection using Python : PSC Assisgnment\t\t",pady=9,bg= 'white',fg='red',justify ="center")
Title_Label.grid(row=0,column=0)
MainFrame = Frame(root,bg = 'white',pady=2,padx=2, width =1450, height = 100, relief = RIDGE)
MainFrame.grid(row=1,column=0)



Label_1 =Label(MainFrame, font=('lato black', 17,'bold'), text="",padx=2,pady=2, fg="black",bg ="white")
Label_1.grid(row=0, column=0)


Label_9 =Button(MainFrame, font=('arial', 17,'bold'), text="  Select Image\n Detect Plate ",padx=2,pady=2, bg="white",fg = "black",command=select_image)
Label_9.grid(row=2, column=0,sticky=W)



    



Label_2 =Label(MainFrame, font=('arial', 10,'bold'), text="\t\t    ",padx=2,pady=2, bg="white",fg = "black")
Label_2.grid(row=3, column=0,sticky=W)

Label_3 =Label(MainFrame, font=('arial', 30,'bold'), text="      \t\t\t",padx=2,pady=2, bg="white",fg = "black")
Label_3.grid(row=4, column=0)

Label_3 =Label(MainFrame, font=('arial', 30,'bold'), text="      \t\t\t",padx=2,pady=2, bg="white",fg = "black")
Label_3.grid(row=5, column=0)

Label_3 =Label(MainFrame, font=('arial', 30,'bold'), text="      \t\t\t",padx=2,pady=2, bg="white",fg = "black")
Label_3.grid(row=6, column=0)

Label_3 =Label(MainFrame, font=('arial', 30,'bold'), text="      \t\t\t",padx=2,pady=2, bg="white",fg = "black")
Label_3.grid(row=7, column=0)

Label_3 =Label(MainFrame, font=('arial', 30,'bold'), text="      \t\t\t",padx=2,pady=2, bg="white",fg = "black")
Label_3.grid(row=8, column=0)

Label_3 =Label(MainFrame, font=('arial', 10,'bold'), text="\t\t\t\t          ",padx=2,pady=2, bg="white",fg = "black")
Label_3.grid(row=9, column=1)





root.mainloop()



#These are the pre-processing steps required to create a haar cascade classifier

import urllib.request
import cv2
import os

def store_raw_images():
    #Collecting non-human image URLS from the website imagenet
    
    neg_images_link = 'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n00007846'
    
    #Reading image URLs
    neg_image_urls = urllib.request.urlopen(neg_images_link).read().decode()
    pic_num = 1
    
    #Creating a folder called neg which will store all the negative images
    if not os.path.exists('neg'):
        os.makedirs('neg')

#Splitting the URLs to process them        
    for i in neg_image_urls.split('\n'):
        try:
            print(i)
            #Retrieving the URL and adding the image into the folder created
            urllib.request.urlretrieve(i, "pos/"+str(pic_num)+".jpg")
            img = cv2.imread("pos/"+str(pic_num)+".jpg",cv2.IMREAD_GRAYSCALE)
            # should be larger than samples / pos pic (so we can place our image on it)
            resized_image = cv2.resize(img, (100, 100))
            #All the images are resized and written into the folder
            cv2.imwrite("pos/"+str(pic_num)+".jpg",resized_image)
            pic_num += 1
            
        except Exception as e:
            print(str(e))  
            


def store_rawpos_images():
    #Collecting human image URLS from the website imagenet
    
    pos_images_link = 'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n00007846'
    
    #Reading image URLs
    pos_image_urls = urllib.request.urlopen(pos_images_link).read().decode()
    pic_num = 1
    
    if not os.path.exists('pos'):
        os.makedirs('pos')
        
    for i in pos_image_urls.split('\n'):
        try:
            print(i)
            #Retrieving the URL and adding the image into the folder created
            urllib.request.urlretrieve(i, "pos/"+str(pic_num)+".jpg")
            img = cv2.imread("pos/"+str(pic_num)+".jpg",cv2.IMREAD_GRAYSCALE)
            # should be larger than samples / pos pic (so we can place our image on it)
            resized_image = cv2.resize(img, (100, 100))
            #All the images are resized and written into the folder
            cv2.imwrite("pos/"+str(pic_num)+".jpg",resized_image)
            pic_num += 1
        #All the exceptions are printed    
        except Exception as e:
            print(str(e)) 
            
            
            
store_raw_images()
store_rawpos_images()


#This function creates a file called bg.txt that stores all the image names so that they can be accessed easily 
def create_pos_n_neg():
    for file_type in ['neg']:
        
        for img in os.listdir(file_type):

            if file_type == 'pos':
                line = file_type+'/'+img+' 1 0 0 50 50\n'
                #info.dat stroes information about positive images
                with open('info.dat','a') as f:
                    f.write(line)
            elif file_type == 'neg':
                line = file_type+'/'+img+'\n'
                with open('bg.txt','a') as f:
                    f.write(line)


import cv2
import matplotlib.pyplot as plt

def get_canny_image(img):
    x = 50
    y = 50
    h = 450
    w = 450
    crop_img = img[y:y+h, x:x+w, :]
    downsize_crop = cv2.resize(crop_img, (64, 64))
    blur = cv2.GaussianBlur(downsize_crop, (5, 5), 0)
    edges = cv2.Canny(blur,50,100)
    plt.imshow(edges, cmap='gray')
    plt.show()
    return edges

def get_image(num):
    try:
        return plt.imread('./binural-updates/ear_photos/Subject_'+str(num)+'/'+str(num)+'_left_side.jpg'), 0
    except:
        try:
            return plt.imread('./binural-updates/ear_photos/Subject_'+str(num)+'/'+str(num)+'_right_side.jpg'), 1
        except:
            try:
                return plt.imread('./binural-updates/ear_photos/Subject_'+str(num)+'/0'+str(num)+'_left.jpg'), 0
            except:
                try:
                    return plt.imread('./binural-updates/ear_photos/Subject_'+str(num)+'/0'+str(num)+'_right.jpg'), 1
                except:
                    try:
                        return plt.imread('./binural-updates/ear_photos/Subject_'+str(num)+'/0'+str(num)+'_left.JPG'), 0
                    except:
                        try:
                            return plt.imread('./binural-updates/ear_photos/Subject_'+str(num)+'/0'+str(num)+'_right.JPG'), 1
                        except:
                            try:
                                return plt.imread('./binural-updates/ear_photos/Subject_'+str(num)+'/Subject_'+str(num)+'_left_side.jpg'), 0
                            except:
                                try:
                                    return plt.imread('./binural-updates/ear_photos/Subject_'+str(num)+'/0'+str(num)+'_left_2.jpg'), 0
                                except:
                                    return plt.imread('./binural-updates/ear_photos/Subject_'+str(num)+'/00'+str(58)+'_left.jpg'), 0
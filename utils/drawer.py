import numpy as np
import cv2


# ============================================================================


FINAL_LINE_COLOR = (255, 255, 255)
WHITE = [255,255,255]

# ============================================================================

class MaskDrawer(object):
    '''Draw a mask on a image'''
    def __init__(self, filename, maskfilename):
        self.filename = filename
        self.maskfilename = maskfilename
        self.drawing = False # Flag signalling that first click of mouse is done
        self.done = False # Flag signalling we're done
        self.points = [] # List of points defining our polygon
        self.img = cv2.imread(self.filename)
        self.img_copy = self.img.copy()
        self.mask = np.zeros(self.img.shape[:2],dtype = np.uint8)
        self.num_images = 0
        self.thickness = int(self.img.shape[0] / 100)

    def reset(self):
        #print("resetting \n")
        self.drawing = False
        self.done = False
        self.img = self.img_copy.copy()
        self.mask = np.zeros(self.img.shape[:2],dtype = np.uint8)
        self.points = []
        cv2.imshow(self.filename,self.img)
        

    def on_mouse(self, event, x, y, buttons, user_param):
        # Mouse callback that gets called for every mouse event (i.e. moving, clicking, etc.)
        global masks_count

        if event == cv2.EVENT_MOUSEMOVE:
            #Store coordinates of mouse movement in array
            if self.drawing == True:
                cv2.circle(self.img,(x,y),self.thickness, WHITE,-1)
                self.points.append((x, y))
            
        elif event == cv2.EVENT_LBUTTONDOWN:
            # Left click starts new mask
            self.reset()
            self.drawing = True
            cv2.circle(self.img,(x,y),self.thickness, WHITE,-1)
            self.points.append((x, y))
            
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing == True:
                self.drawing = False
                self.done = True
                cv2.circle(self.img,(x,y),self.thickness, WHITE,-1)
                self.points.append((x, y))
                if (len(self.points) > 0):
                    cv2.fillPoly(self.mask, np.array([self.points]), FINAL_LINE_COLOR)
                    overlay = self.img.copy()
                    cv2.fillPoly(overlay, np.array([self.points]), FINAL_LINE_COLOR)
                self.img = cv2.addWeighted(overlay, 0.5, self.img, 0.5, 0.0, dtype=cv2.CV_8UC3)
                cv2.imshow(self.filename, self.img) 
                print('Press s to continue and save the drawing')
                print('Press q to move to next image without saving')
                print('Press esc to exit the drawing tool')

    def run(self):
        # Open image in a window and set a mouse callback to handle events
        cv2.namedWindow(self.filename, cv2.WINDOW_NORMAL)
        self.img_copy = self.img.copy()  # a copy of original image
        self.mask = np.zeros(self.img.shape[:2],dtype = np.uint8) # mask initialized to black image
        cv2.imshow(self.filename,self.img)
        print('To localize the pattern, draw a line around it. Press s to continue')
        print('If no localization is required, press s to continue')
        print('Press q to move to next image without saving')
        print('Press esc to exit the drawing tool')
        
        cv2.waitKey(1)
        cv2.setMouseCallback(self.filename, self.on_mouse)

        while(True):
            cv2.imshow(self.filename,self.img)
            k = cv2.waitKey(1)
            # key bindings
            if k == ord('s'):
                if self.done:
                    cv2.imwrite(self.maskfilename, self.mask)
                    print('Mask is captured and saved to {}'.format(self.maskfilename))
                else:
                    print('Nothing to save. Moving to next image.')
                cv2.destroyWindow(self.filename)
                return 'save'
                
            if k == ord('q'):
                print('Mask is not saved')
                cv2.destroyWindow(self.filename)
                return 'next'
                
            if k == 27:
                print('Exit the drawing tool')
                cv2.destroyWindow(self.filename)
                return 'exit'
         
        
        
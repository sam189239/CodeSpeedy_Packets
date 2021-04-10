import matplotlib
from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN

dir = 'H:\\Code\\current_files\\My_Code\\Face_Recognition'

def draw_Boxes(filename,box_list):
    img = pyplot.imread(filename)
    pyplot.imshow(img)
    context = pyplot.gca()
    for box in box_list:
        x,y,width,height = box['box']
        rect = matplotlib.patches.Rectangle((x,y), width, height, fill = False, color = 'blue')
        context.add_patch(rect)
    pyplot.show()

def draw_Keypoints(filename, box_list):
    img = pyplot.imread(filename)
    pyplot.imshow(img)
    context = pyplot.gca()
    for box in box_list:
        for key, value in box['keypoints'].items():
            circle = matplotlib.patches.Circle(value, radius = 1, color = 'red')
            context.add_patch(circle)
    pyplot.show()

def draw_Box_and_Keypoints(filename, box_list):
    img = pyplot.imread(filename)
    pyplot.imshow(img)
    context = pyplot.gca()
    for box in box_list:
        for key, value in box['keypoints'].items():
            circle = matplotlib.patches.Circle(value, radius = 1, color = 'red')
            x,y,width,height = box['box']
            rect = matplotlib.patches.Rectangle((x,y), width, height, fill = False, color = 'blue')
            context.add_patch(rect)
            context.add_patch(circle)
    pyplot.show()

def extract_Faces(filename, box_list):
    img = pyplot.imread(filename)
    pyplot.imshow(img)
    for i in range(len(box_list)):
        x1, y1, width, height = box_list[i]['box']
        x2, y2 = x1 + width, y1 + height
        face = img[y1:y2, x1:x2]
        pyplot.subplot(1,len(box_list),i+1)
        pyplot.axis('off')
        pyplot.imshow(face)
    pyplot.show()


filename = dir + '/123.jpg'
img = pyplot.imread(filename)
detector = MTCNN()
faces = detector.detect_faces(img)
for face in faces:
    print(face)
draw_Boxes(filename, faces)
draw_Keypoints(filename, faces)
draw_Box_and_Keypoints(filename, faces)
extract_Faces(filename, faces)

# import required module
import os
import xml.etree.ElementTree as ET
class_to_nr = {"D00": 0, "D10": 1, "D20": 2, "D40": 3}

def convert_annotations_to_YOLO(folderpath):
    filelist = []
    # iterate over files in
    # that directory
    xml_path = folderpath+"/annotations/xmls"
    for filename in os.listdir(xml_path):
        f = os.path.join(xml_path, filename)
        # checking if it is a file
        if os.path.isfile(f):
            filelist.append(filename)

    for filename in filelist:
        tree = ET.parse(xml_path+"/"+filename)
        root = tree.getroot()

        f_name = root.findtext("filename").split(".")[0]
        width = float(root.findall("size")[0].findtext("width"))
        height = float(root.findall("size")[0].findtext("height"))
        
        objects = root.findall("object")
        if objects == []:
            continue
        file =open(folderpath+"/labels/"+f_name+".txt", "w" )
        for obj in objects:
            class_name = obj.findtext("name")
            if class_name not in class_to_nr:
                continue
            class_nr = class_to_nr[class_name]


            bbox = obj.find("bndbox")
            x_min = float(bbox.findtext("xmin"))/width
            x_max = float(bbox.findtext("xmax"))/width
            y_min = float(bbox.findtext("ymin"))/height
            y_max = float(bbox.findtext("ymax"))/height

            box_width = x_max-x_min
            box_height = y_max - y_min

            x_center = x_min + box_width/2
            y_center = y_min + box_height/2

            file.write(f"{class_nr} {x_center} {y_center} {box_width} {box_height}/n")
        file.close()
            



        

prefix = "C:/Users/einar/Desktop/datvis_project/RDD2022_dataset/RDD2022_released_through_CRDDC2022/RDD2022/"
path = prefix+"China_Drone/train"
#convert_annotations_to_YOLO(path)
path = prefix+"China_MotorBike/train"
#convert_annotations_to_YOLO(path)
path = prefix+"Czech/train"
convert_annotations_to_YOLO(path)
path = prefix+"India/train"
convert_annotations_to_YOLO(path)
path = prefix+"Japan/train"
convert_annotations_to_YOLO(path)
path = prefix+"United_States/train"
convert_annotations_to_YOLO(path)
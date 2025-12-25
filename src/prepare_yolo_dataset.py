import os
import xml.etree.ElementTree as ET

IMG_SIZE = 416
classes = ["with_mask", "without_mask", "mask_weared_incorrect"]

def convert_bbox(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    x *= dw
    w *= dw
    y *= dh
    h *= dh
    return x, y, w, h

annotations = "dataset/annotations"
images = "dataset/images"
labels_out = "dataset/yolo_labels"

os.makedirs(labels_out, exist_ok=True)

for file in os.listdir(annotations):
    if not file.endswith(".xml"):
        continue

    tree = ET.parse(os.path.join(annotations, file))
    root = tree.getroot()

    img_w = int(root.find("size/width").text)
    img_h = int(root.find("size/height").text)

    label_file = os.path.join(labels_out, file.replace(".xml", ".txt"))
    f = open(label_file, "w")

    for obj in root.iter("object"):
        cls = obj.find("name").text
        cls_id = classes.index(cls)

        xml_box = obj.find("bndbox")
        xmin = float(xml_box.find("xmin").text)
        ymin = float(xml_box.find("ymin").text)
        xmax = float(xml_box.find("xmax").text)
        ymax = float(xml_box.find("ymax").text)

        x, y, w, h = convert_bbox((img_w, img_h), (xmin, ymin, xmax, ymax))
        f.write(f"{cls_id} {x} {y} {w} {h}\n")

    f.close()

print("YOLO labels created successfully 🚀")

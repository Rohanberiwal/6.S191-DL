
import json
import json
import os
import json
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import os
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
non_mitotic_json = {"A00_03.jpg1286775":{"filename":"A00_03.jpg","size":1286775,"regions":[{"shape_attributes":{"name":"rect","x":491,"y":296,"width":134,"height":128},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":756,"y":317,"width":149,"height":134},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":1127,"y":308,"width":152,"height":143},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":1213,"y":658,"width":158,"height":134},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":1380,"y":987,"width":161,"height":131},"region_attributes":{}}],"file_attributes":{}},"A00_05.jpg1253910":{"filename":"A00_05.jpg","size":1253910,"regions":[{"shape_attributes":{"name":"rect","x":171,"y":332,"width":137,"height":134},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":655,"y":384,"width":171,"height":143},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":768,"y":707,"width":174,"height":152},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":874,"y":1210,"width":168,"height":146},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":1426,"y":728,"width":174,"height":146},"region_attributes":{}}],"file_attributes":{}},"A00_06.jpg1241431":{"filename":"A00_06.jpg","size":1241431,"regions":[{"shape_attributes":{"name":"rect","x":463,"y":676,"width":161,"height":155},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":859,"y":661,"width":161,"height":149},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":1082,"y":771,"width":180,"height":158},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":1127,"y":232,"width":186,"height":165},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":713,"y":1219,"width":189,"height":189},"region_attributes":{}}],"file_attributes":{}},"A00_07.jpg1299203":{"filename":"A00_07.jpg","size":1299203,"regions":[{"shape_attributes":{"name":"rect","x":588,"y":884,"width":180,"height":161},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":1274,"y":951,"width":183,"height":158},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":1441,"y":1423,"width":158,"height":161},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":701,"y":1481,"width":186,"height":161},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":168,"y":1127,"width":174,"height":152},"region_attributes":{}}],"file_attributes":{}},"A00_09.jpg1348163":{"filename":"A00_09.jpg","size":1348163,"regions":[{"shape_attributes":{"name":"rect","x":753,"y":484,"width":177,"height":168},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":1237,"y":615,"width":192,"height":174},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":1347,"y":1191,"width":183,"height":183},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":390,"y":1155,"width":177,"height":165},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":798,"y":1529,"width":171,"height":177},"region_attributes":{}}],"file_attributes":{}},"A01_00.jpg1294895":{"filename":"A01_00.jpg","size":1294895,"regions":[{"shape_attributes":{"name":"rect","x":750,"y":862,"width":168,"height":171},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":1301,"y":896,"width":165,"height":168},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":1472,"y":289,"width":174,"height":177},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":1082,"y":1447,"width":183,"height":171},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":527,"y":1417,"width":189,"height":171},"region_attributes":{}}],"file_attributes":{}},"A01_01.jpg1292277":{"filename":"A01_01.jpg","size":1292277,"regions":[{"shape_attributes":{"name":"rect","x":506,"y":850,"width":155,"height":158},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":932,"y":332,"width":161,"height":158},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":506,"y":384,"width":180,"height":171},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":207,"y":171,"width":158,"height":146},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":1295,"y":91,"width":161,"height":165},"region_attributes":{}}],"file_attributes":{}},"A01_02.jpg1273830":{"filename":"A01_02.jpg","size":1273830,"regions":[{"shape_attributes":{"name":"rect","x":966,"y":1008,"width":189,"height":161},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":198,"y":707,"width":161,"height":183},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":207,"y":1310,"width":177,"height":168},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":558,"y":232,"width":180,"height":152},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":957,"y":1645,"width":174,"height":189},"region_attributes":{}}],"file_attributes":{}},"A01_03.jpg1255845":{"filename":"A01_03.jpg","size":1255845,"regions":[{"shape_attributes":{"name":"rect","x":856,"y":1307,"width":192,"height":183},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":1170,"y":1779,"width":195,"height":171},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":424,"y":1785,"width":189,"height":165},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":457,"y":1274,"width":192,"height":171},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":329,"y":530,"width":180,"height":174},"region_attributes":{}}],"file_attributes":{}},"A01_07.jpg1286657":{"filename":"A01_07.jpg","size":1286657,"regions":[{"shape_attributes":{"name":"rect","x":524,"y":765,"width":165,"height":149},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":1039,"y":826,"width":168,"height":149},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":189,"y":1139,"width":174,"height":155},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":692,"y":1255,"width":171,"height":155},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":1136,"y":1670,"width":174,"height":183},"region_attributes":{}}],"file_attributes":{}},"A01_08.jpg1293717":{"filename":"A01_08.jpg","size":1293717,"regions":[{"shape_attributes":{"name":"rect","x":646,"y":1660,"width":177,"height":174},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":1136,"y":1569,"width":195,"height":183},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":1673,"y":1344,"width":183,"height":161},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":1036,"y":1255,"width":177,"height":180},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":1615,"y":1837,"width":192,"height":186},"region_attributes":{}}],"file_attributes":{}},"A02_02.jpg1380812":{"filename":"A02_02.jpg","size":1380812,"regions":[{"shape_attributes":{"name":"rect","x":868,"y":1176,"width":225,"height":247},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":1475,"y":1121,"width":180,"height":207},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":1234,"y":1621,"width":195,"height":174},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":698,"y":1825,"width":186,"height":180},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":1179,"y":451,"width":174,"height":186},"region_attributes":{}}],"file_attributes":{}},"A02_04.jpg1327830":{"filename":"A02_04.jpg","size":1327830,"regions":[{"shape_attributes":{"name":"rect","x":963,"y":1161,"width":186,"height":186},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":1088,"y":710,"width":171,"height":177},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":411,"y":917,"width":158,"height":158},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":518,"y":1365,"width":192,"height":186},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":1350,"y":1514,"width":210,"height":195},"region_attributes":{}}],"file_attributes":{}},"A02_05.jpg1404220":{"filename":"A02_05.jpg","size":1404220,"regions":[{"shape_attributes":{"name":"rect","x":1027,"y":1176,"width":183,"height":161},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":1389,"y":1499,"width":229,"height":201},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":628,"y":1648,"width":152,"height":152},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":158,"y":1624,"width":155,"height":165},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":466,"y":481,"width":186,"height":168},"region_attributes":{}}],"file_attributes":{}},"A02_06.jpg1391554":{"filename":"A02_06.jpg","size":1391554,"regions":[{"shape_attributes":{"name":"rect","x":902,"y":1411,"width":158,"height":155},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":1237,"y":990,"width":180,"height":158},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":1603,"y":1523,"width":177,"height":152},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":1368,"y":1871,"width":183,"height":165},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":670,"y":1868,"width":180,"height":165},"region_attributes":{}}],"file_attributes":{}},"A02_08.jpg1371045":{"filename":"A02_08.jpg","size":1371045,"regions":[{"shape_attributes":{"name":"rect","x":229,"y":966,"width":192,"height":192},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":497,"y":588,"width":207,"height":213},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":1015,"y":606,"width":207,"height":201},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":494,"y":116,"width":219,"height":207},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":954,"y":192,"width":247,"height":216},"region_attributes":{}}],"file_attributes":{}},"A02_09.jpg1355082":{"filename":"A02_09.jpg","size":1355082,"regions":[{"shape_attributes":{"name":"rect","x":289,"y":679,"width":177,"height":189},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":731,"y":759,"width":210,"height":225},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":1115,"y":859,"width":204,"height":189},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":1554,"y":463,"width":174,"height":174},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":954,"y":527,"width":195,"height":189},"region_attributes":{}}],"file_attributes":{}},"A03_02.jpg1362615":{"filename":"A03_02.jpg","size":1362615,"regions":[{"shape_attributes":{"name":"rect","x":1429,"y":743,"width":165,"height":174},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":1673,"y":207,"width":183,"height":180},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":1237,"y":225,"width":183,"height":171},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":1834,"y":609,"width":186,"height":171},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":1459,"y":1136,"width":201,"height":180},"region_attributes":{}}],"file_attributes":{}},"A03_03.jpg1467799":{"filename":"A03_03.jpg","size":1467799,"regions":[{"shape_attributes":{"name":"rect","x":530,"y":1219,"width":165,"height":165},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":238,"y":920,"width":155,"height":143},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":317,"y":1255,"width":158,"height":171},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":442,"y":1712,"width":180,"height":195},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":107,"y":1581,"width":183,"height":183},"region_attributes":{}}],"file_attributes":{}},"A03_05.jpg1442474":{"filename":"A03_05.jpg","size":1442474,"regions":[{"shape_attributes":{"name":"rect","x":597,"y":941,"width":161,"height":171},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":996,"y":941,"width":204,"height":183},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":1167,"y":497,"width":177,"height":158},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":1459,"y":250,"width":183,"height":168},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":1600,"y":606,"width":183,"height":189},"region_attributes":{}}],"file_attributes":{}},"A03_06.jpg1411210":{"filename":"A03_06.jpg","size":1411210,"regions":[{"shape_attributes":{"name":"rect","x":1615,"y":1785,"width":247,"height":229},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":871,"y":1855,"width":158,"height":177},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":1155,"y":1673,"width":183,"height":186},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":344,"y":1639,"width":192,"height":180},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":817,"y":1441,"width":183,"height":189},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":1280,"y":1277,"width":207,"height":195},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":372,"y":399,"width":149,"height":161},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":52,"y":503,"width":177,"height":168},"region_attributes":{}}],"file_attributes":{}},"A03_07.jpg1437906":{"filename":"A03_07.jpg","size":1437906,"regions":[{"shape_attributes":{"name":"rect","x":859,"y":1054,"width":183,"height":195},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":1222,"y":835,"width":198,"height":198},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":1627,"y":813,"width":219,"height":210},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":838,"y":512,"width":189,"height":180},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":591,"y":104,"width":177,"height":183},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":917,"y":174,"width":225,"height":195},"region_attributes":{}}],"file_attributes":{}},"A03_08.jpg1401233":{"filename":"A03_08.jpg","size":1401233,"regions":[{"shape_attributes":{"name":"rect","x":1557,"y":1481,"width":198,"height":180},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":1627,"y":1106,"width":165,"height":155},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":1243,"y":1362,"width":204,"height":204},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":1185,"y":1758,"width":235,"height":198},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":795,"y":1855,"width":192,"height":177},"region_attributes":{}}],"file_attributes":{}},"A03_09.jpg1387683":{"filename":"A03_09.jpg","size":1387683,"regions":[{"shape_attributes":{"name":"rect","x":914,"y":356,"width":186,"height":171},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":1289,"y":24,"width":213,"height":189},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":274,"y":152,"width":204,"height":177},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":631,"y":110,"width":192,"height":171},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":487,"y":494,"width":229,"height":195},"region_attributes":{}}],"file_attributes":{}},"A04_00.jpg1353403":{"filename":"A04_00.jpg","size":1353403,"regions":[{"shape_attributes":{"name":"rect","x":1526,"y":798,"width":171,"height":177},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":1807,"y":795,"width":189,"height":174},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":1749,"y":448,"width":180,"height":174},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":1380,"y":460,"width":177,"height":174},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":1590,"y":158,"width":161,"height":158},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":1018,"y":902,"width":171,"height":149},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":1405,"y":1143,"width":180,"height":174},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":1731,"y":1155,"width":189,"height":174},"region_attributes":{}}],"file_attributes":{}},"A04_01.jpg1387243":{"filename":"A04_01.jpg","size":1387243,"regions":[{"shape_attributes":{"name":"rect","x":1593,"y":219,"width":183,"height":171},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":1170,"y":186,"width":174,"height":158},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":1313,"y":548,"width":210,"height":207},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":1779,"y":585,"width":201,"height":207},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":676,"y":695,"width":204,"height":195},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":695,"y":161,"width":180,"height":186},"region_attributes":{}}],"file_attributes":{}},"A04_02.jpg1387978":{"filename":"A04_02.jpg","size":1387978,"regions":[{"shape_attributes":{"name":"rect","x":877,"y":1036,"width":238,"height":213},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":743,"y":817,"width":180,"height":149},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":466,"y":1048,"width":216,"height":204},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":158,"y":1304,"width":192,"height":204},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":210,"y":661,"width":213,"height":186},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":323,"y":119,"width":225,"height":213},"region_attributes":{}}],"file_attributes":{}},"A04_05.jpg1412859":{"filename":"A04_05.jpg","size":1412859,"regions":[{"shape_attributes":{"name":"rect","x":37,"y":1749,"width":146,"height":174},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":274,"y":1859,"width":198,"height":161},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":728,"y":1883,"width":183,"height":171},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":1173,"y":1868,"width":216,"height":180},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":1563,"y":1813,"width":201,"height":216},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":1764,"y":1368,"width":219,"height":201},"region_attributes":{}}],"file_attributes":{}},"A04_06.jpg1372595":{"filename":"A04_06.jpg","size":1372595,"regions":[{"shape_attributes":{"name":"rect","x":277,"y":1200,"width":229,"height":201},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":676,"y":996,"width":195,"height":192},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":725,"y":1344,"width":219,"height":204},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":430,"y":1694,"width":210,"height":198},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":756,"y":1688,"width":219,"height":198},"region_attributes":{}}],"file_attributes":{}},"A04_08.jpg1403465":{"filename":"A04_08.jpg","size":1403465,"regions":[{"shape_attributes":{"name":"rect","x":417,"y":972,"width":198,"height":207},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":628,"y":606,"width":232,"height":210},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":241,"y":274,"width":198,"height":201},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":743,"y":225,"width":229,"height":225},"region_attributes":{}},{"shape_attributes":{"name":"rect","x":1149,"y":652,"width":229,"height":207},"region_attributes":{}}],"file_attributes":{}}}
saver_dict =  {}
directory = r"C:\Users\rohan\OneDrive\Desktop\Tester"
base_model = VGG16(weights='imagenet', include_top=False)
model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_conv3').output)
image_folder =  r"C:\Users\rohan\OneDrive\Desktop\Tester"
path = r"C:\Users\rohan\OneDrive\Desktop\Training"
covered_set = set()
def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def extract_feature_map(image, bbox):
    x, y, w, h = bbox
    crop_img = image[y:y+h, x:x+w]
    crop_img = cv2.resize(crop_img, (224, 224))
    crop_img = np.expand_dims(crop_img, axis=0)
    crop_img = preprocess_input(crop_img)
    features = model.predict(crop_img)
    return features

def plot_feature_map(feature_map, bbox_coords):
    fig, axes = plt.subplots(8, 8, figsize=(15, 15))
    for i, ax in enumerate(axes.flat):
        if i < feature_map.shape[-1]:
            ax.imshow(feature_map[0, :, :, i], cmap='viridis')
        ax.axis('off')
    fig.suptitle(f'Feature Maps for Bounding Box {bbox_coords}')
    plt.show()

def annotate(saver_dict, path):
    for filename, regions in saver_dict.items():
        image_path = os.path.join(path, filename)
        
        if os.path.exists(image_path):
            try:
                with Image.open(image_path) as img:
                    draw = ImageDraw.Draw(img)
                    for region in regions:
                        x, y, width, height = region  
                        box = [x, y, x + width, y + height]
                        draw.rectangle(box, outline="purple", width=3)
                        
                    #img.show()
                        
                    for region in regions:
                        print("file name is ",filename)
                        x, y, width, height = region  # Unpack coordinates
                        box = [x, y, x + width, y + height]
                        draw.rectangle(box, outline="purple", width=3)
                        
                        image = np.array(img)
                        # Extract features for the region
                        features = extract_feature_map(image, region)
                        
                        print(features)
                        
                    
                    # Save annotated image with a new filename
                    save_path = os.path.join(path, f"annotated_{filename}")
                    img.save(save_path)
                    
            except Exception as e:
                print(f"Error processing {filename}: {e}")
        else:
            print(f"File {filename} does not exist in the specified folder.")


"""
def extract_feature_map(image, bbox):
    x, y, w, h = bbox
    crop_img = image[y:y+h, x:x+w]
    crop_img = cv2.resize(crop_img, (224, 224))
    crop_img = np.expand_dims(crop_img, axis=0)
    crop_img = preprocess_input(crop_img)
    features = model.predict(crop_img)
    return features

def plot_feature_map(feature_map, bbox_coords):
    fig, axes = plt.subplots(8, 8, figsize=(15, 15))
    for i, ax in enumerate(axes.flat):
        if i < feature_map.shape[-1]:
            ax.imshow(feature_map[0, :, :, i], cmap='viridis')
        ax.axis('off')
    fig.suptitle(f'Feature Maps for Bounding Box {bbox_coords}')
    plt.show()

def annotate(saver_dict, path):
    for filename, regions in saver_dict.items():
        image_path = os.path.join(path, filename)
        
        if os.path.exists(image_path):
            try:
                with Image.open(image_path) as img:
                    draw = ImageDraw.Draw(img)
                    for region in regions:
                        x, y, width, height = region 
                        box = [x, y, x + width, y + height]
                        draw.rectangle(box, outline="purple", width=3)
                        

                    #img.show()
                    
                    for region in regions :
                        x, y, width, height = region 
                        box = [x, y, x + width, y + height]
                        file_image = os.path.join(path , filename)
                        print(file_image)
                        covered_set.add(file_image)
                        var = extract_feature_map(file_image , region)
                        print("Maps for the Boundary box  ",box, " " , var)
                        
                    save_path = os.path.join(path, f"annotated_{filename}")
                    img.save(save_path)
                    
            except Exception as e:
                print(f"Error processing {filename}: {e}")
        else:
            print(f"File {filename} does not exist in the specified folder.")

"""

def print_filename_bbox(json_file):
    universal_list  = []
    with open(json_file, 'r') as f:
        data = json.load(f)
    for image_key, image_data in data.items():
        filename = image_data.get('filename', 'Unknown')
        print(f"File Name: {filename}")
        boundary_box =  []
        for region in image_data.get('regions', []):
            shape_attributes = region.get('shape_attributes', {})
            xmin = shape_attributes.get('x', 'N/A')
            ymin = shape_attributes.get('y', 'N/A')
            width = shape_attributes.get('width', 'N/A')
            height = shape_attributes.get('height', 'N/A')
            
            print(f"Bounding Box Coordinates: xmin={xmin}, ymin={ymin}, width={width}, height={height}")
            boundary_box.append(xmin)
            boundary_box.append(ymin)
            boundary_box.append(width)
            boundary_box.append(height)
            universal_list.append(boundary_box)
            boundary_box =  []
        print("------------------------")
        saver_dict[filename] = universal_list
        universal_list = []
        boundary_box = []     
    print(universal_list)
    return saver_dict

json_file = 'NonMitotic.json'
nums = print_filename_bbox(json_file)
print(nums)
annotate(nums , path)

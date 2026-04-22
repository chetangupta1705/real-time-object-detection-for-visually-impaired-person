import os

image_folder = "dataset/images"
label_folder = "dataset/labels"

images = sorted(os.listdir(image_folder))
labels = sorted(os.listdir(label_folder))

print("Images:", len(images))
print("Labels:", len(labels))

for i in range(min(len(images), len(labels))):
    
    img_old = os.path.join(image_folder, images[i])
    lbl_old = os.path.join(label_folder, labels[i])

    new_name = f"img_{i}.jpg"
    new_lbl = f"img_{i}.txt"

    os.rename(img_old, os.path.join(image_folder, new_name))
    os.rename(lbl_old, os.path.join(label_folder, new_lbl))

print("Matching Done ✅")
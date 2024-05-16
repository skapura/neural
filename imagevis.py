from keras.utils import image_dataset_from_directory
from keras import layers, models
import numpy as np
import mlutil
import cv2


trainds = image_dataset_from_directory('images_large/train', labels='inferred', label_mode='categorical', shuffle=False, image_size=(256, 256))
valds = image_dataset_from_directory('images_large/val', labels='inferred', label_mode='categorical', shuffle=False, image_size=(256, 256))
#trainds = trainds.prefetch(tf_data.AUTOTUNE)
#valds = valds.prefetch(tf_data.AUTOTUNE)
model = models.load_model('largeimage.keras', compile=True)

batchindex = 0
for imagebatch, labelbatch in valds:
    outs = model.predict(imagebatch)
    for i in range(len(imagebatch)):
        a = imagebatch[i, :, :, :]
        img = cv2.imread(valds.file_paths[batchindex])
        rimg = cv2.resize(img, (256, 256))
        pred = np.argmax(outs[i])
        h, himg = mlutil.heatmap(np.asarray([imagebatch[i, :, :, :]]), model, 'activation_3', pred)
        a = imagebatch[i, :, :, :]
        heatout = mlutil.overlayHeatmap(np.asarray([rimg]), h)
        cv2.imwrite('.session/heat_' + str(batchindex).zfill(5) + '.png', heatout)
        cv2.imwrite('.session/heat_' + str(batchindex).zfill(5) + '_orig.png', rimg)
        batchindex += 1
        print(i)

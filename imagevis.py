from keras.utils import image_dataset_from_directory
from keras import layers, models
import numpy as np
import mlutil
import cv2
import pickle


trainds = image_dataset_from_directory('images_large/train', labels='inferred', label_mode='categorical', shuffle=False, image_size=(256, 256))
valds = image_dataset_from_directory('images_large/val', labels='inferred', label_mode='categorical', shuffle=False, image_size=(256, 256))
layernames = ['activation', 'activation_1', 'activation_2', 'activation_3', 'prediction']
model = models.load_model('largeimage.keras', compile=True)
layermodel = mlutil.makeLayerOutputModel(model, layernames)

franges = None
#for imagebatch, labelbatch in valds:
#    franges = mlutil.getLayerOutputRange(model, layernames, imagebatch, franges)
#with open('franges.pkl', 'wb') as f:
#    pickle.dump(franges, f)

with open('franges.pkl', 'rb') as f:
    franges = pickle.load(f)

batchindex = 0
for imagebatch, labelbatch in valds:
    outs = layermodel.predict(imagebatch)
    for i in range(len(imagebatch)):
        li = 0
        for oi in range(len(outs) - 1):
            for fi in range(outs[oi].shape[-1]):
                if fi == 9:
                    print(1)
                fmap = outs[oi][i, :, :, fi]
                f_min, f_max = franges[oi][fi][0], franges[oi][fi][1]
                if f_max == 0.0:
                    scaled = fmap
                else:
                    scaled = np.interp(fmap, [f_min, f_max], [0, 255]).astype(np.uint8)
                filename = 'feature_' + str(batchindex).zfill(5) + '_' + str(li).zfill(2) + '_' + str(fi).zfill(3) + '.png'
                cv2.imwrite('.session/features/' + filename, scaled)
            li += 1
        break
        batchindex += 1
print(1)


# Generate heatmaps
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

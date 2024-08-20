from tensorflow.keras.applications.xception import Xception, preprocess_input, decode_predictions
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing import image
import numpy as np


model = Xception(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
)

#image1 = image.load_img('imagenette2-320/train/chain_saw/ILSVRC2012_val_00024269.JPEG', target_size=(299, 299))
#image1 = image.load_img('imagenette2-320/train/church/ILSVRC2012_val_00016949.JPEG', target_size=(299, 299)) # church
#image1 = image.load_img('imagenette2-320/train/golf_ball/ILSVRC2012_val_00019268.JPEG', target_size=(299, 299))
#image1 = image.load_img('imagenette2-320/train/dog/ILSVRC2012_val_00046295.JPEG', target_size=(299, 299))
#image1 = image.load_img('imagenette2-320/train/french_horn/n03394916_1769.JPEG', target_size=(299, 299))
#image1 = image.load_img('imagenette2-320/train/n03417042/ILSVRC2012_val_00005094.JPEG', target_size=(299, 299))
#image1 = image.load_img('imagenette2-320/train/gas_pump/ILSVRC2012_val_00017469.JPEG', target_size=(299, 299))
image1 = image.load_img('imagenette2-320/train/parachute/ILSVRC2012_val_00004745.JPEG', target_size=(299, 299))

transformedImage = image.img_to_array(image1)
transformedImage = np.expand_dims(transformedImage, axis=0)
transformedImage = preprocess_input(transformedImage)
prediction = model.predict(transformedImage)
predictionLabel = decode_predictions(prediction, top=15)
print(predictionLabel)

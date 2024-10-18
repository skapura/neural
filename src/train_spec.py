from keras import models
import data
import sys

if __name__ == '__main__':
    epochs = int(sys.argv[1])
    trainds = data.load_from_directory('datasets/images_large/train')
    model = models.load_model('session/temp_model.keras', compile=True)
    model.fit(trainds, epochs=epochs)
    model.save('session/temp_model.keras')

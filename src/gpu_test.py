import tensorflow as tf


print(tf.config.list_physical_devices())
print('GPU Support: ' + str(tf.test.is_built_with_gpu_support()))
print('CUDA: ' + str(tf.test.is_built_with_cuda()))
print('TensorFlow version: ' + tf.version.VERSION)

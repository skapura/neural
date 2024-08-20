model = VGG16()
print(model.summary())

#img = load_img('bird.jpg', target_size=(224, 224))
img = load_img('goose.jpeg', target_size=(224, 224))
img = img_to_array(img)
img = np.expand_dims(img, axis=0)

debugmodel = makeDebugModel(model)
print(backend.image_data_format())
filters, bias = debugmodel.layers[1].get_weights()
#renderFilters(filters)

outs = debugmodel.predict(img)

#outs = debugmodel.predict(np.asarray([test_images[0]]))
layer1 = outs[1][0]
renderFeatureMaps(layer1)
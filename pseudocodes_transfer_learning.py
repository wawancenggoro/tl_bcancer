# conv init, block1, block2
model.densenet121.features0[4][0][2].weight = model_ori.densenet121.features[4][0][2].weight
model.densenet121.features0[4][0][2].weight.require_grad = False

model.densenet121.features1[4][0][2].weight = model_ori.densenet121.features[4][0][2].weight
model.densenet121.features1[4][0][2].weight.require_grad = False

model.densenet121.features2[4][0][2].weight = model_ori.densenet121.features[4][0][2].weight
model.densenet121.features2[4][0][2].weight.require_grad = False

model.densenet121.features3[4][0][2].weight = model_ori.densenet121.features[4][0][2].weight
model.densenet121.features3[4][0][2].weight.require_grad = False

# block3, block4
model.densenet121.features[1][0][2].weight = model_ori.densenet121.features[6][0][2].weight
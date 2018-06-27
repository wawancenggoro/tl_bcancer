x0 = torch.autograd.Variable(torch.from_numpy(np.uint8(np.random.uniform(150, 180, (1, 3, 224, 224)))).float()).cuda()
x0 = model.module.densenet121.features0.conv0(x0)
x0 = model.module.densenet121.features0.pool0(x0)
x0 = model.module.densenet121.features0.denseblock1(x0)
x0 = model.module.densenet121.features0.transition1(x0)
x0 = model.module.densenet121.features0.denseblock2(x0)

x1 = torch.autograd.Variable(torch.from_numpy(np.uint8(np.random.uniform(150, 180, (1, 3, 224, 224)))).float()).cuda()
x1 = model.module.densenet121.features0.conv0(x1)
x1 = model.module.densenet121.features0.pool0(x1)
x1 = model.module.densenet121.features0.denseblock1(x1)
x1 = model.module.densenet121.features0.transition1(x1)
x1 = model.module.densenet121.features0.denseblock2(x1)

x2 = torch.autograd.Variable(torch.from_numpy(np.uint8(np.random.uniform(150, 180, (1, 3, 224, 224)))).float()).cuda()
x2 = model.module.densenet121.features0.conv0(x2)
x2 = model.module.densenet121.features0.pool0(x2)
x2 = model.module.densenet121.features0.denseblock1(x2)
x2 = model.module.densenet121.features0.transition1(x2)
x2 = model.module.densenet121.features0.denseblock2(x2)

x3 = torch.autograd.Variable(torch.from_numpy(np.uint8(np.random.uniform(150, 180, (1, 3, 224, 224)))).float()).cuda()
x3 = model.module.densenet121.features0.conv0(x3)
x3 = model.module.densenet121.features0.pool0(x3)
x3 = model.module.densenet121.features0.denseblock1(x3)
x3 = model.module.densenet121.features0.transition1(x3)
x3 = model.module.densenet121.features0.denseblock2(x3)

x = torch.cat([x0, x1, x2, x3], 1)
x = model.module.densenet121.features.bottleneck(x)

x = model.module.densenet121.features.denseblock3(x)
x = model.module.densenet121.features.transition3(x)
x = model.module.densenet121.features.denseblock4(x)
x = model.module.densenet121.features.norm5(x)



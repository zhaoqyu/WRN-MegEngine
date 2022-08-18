import megengine as mge

model = mge.hub.load("zhaoqyu/WRN-MegEngine", "wide_resnet50_2", git_host='github.com', use_cache=False, pretrained=True)

print(model)

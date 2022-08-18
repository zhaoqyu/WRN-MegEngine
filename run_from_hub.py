import megengine as mge

model = mge.hub.load("zhaoqyu/openpose-mge-pt", "openpose_model", git_host='github.com', pretrained=True)

print(model)

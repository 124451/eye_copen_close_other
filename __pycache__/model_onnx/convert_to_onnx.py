import torch
from mixnet import MixNet

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def load_model(model, pretrained_path):
    # 参数保存在cpu上
    pretrained_dict = torch.load(pretrained_path,map_location=lambda storage, loc:storage)

    pretrained_dict = remove_prefix(pretrained_dict,'module.')
    check_keys(model=model,pretrained_state_dict=pretrained_dict)
    model.load_state_dict(pretrained_dict,strict=True)
    return model



def main():
    model_path = "/media/omnisky/D4T/JSH/faceFenlei/Projects/hul_eye_class/weight/relabel_04_mix_SGD_mutillabel_24_24_20210302/Mixnet_epoch_49.pth"
    # model
    mixnet = MixNet(input_size=(24,24),num_classes=3)
    net = load_model(mixnet,pretrained_path=model_path)
    net = net.to("cpu")
    # export
    output_onnx = "/media/omnisky/D4T/JSH/faceFenlei/Projects/hul_eye_class/__pycache__/model_onnx/eyeclassification_softmax.onnx"
    input_names = ["input0"]
    output_names = ["output0"]
    inputs = torch.randn(1,3,24,24).to("cpu")
    torch_out = torch.onnx._export(net, inputs, output_onnx, export_params=True, verbose=False,
                                   input_names=input_names, output_names=output_names,opset_version=11)
    


if __name__ == "__main__":
    main()
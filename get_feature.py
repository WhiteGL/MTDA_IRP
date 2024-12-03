import os
import numpy as np
import torch.backends.cudnn as cudnn
import torch.utils.data
from data_loader import get_data_loader

def test(dataset_name, model_name):
    assert dataset_name in ['source', 'target']

    cuda = True
    cudnn.benchmark = True
    batch_size = 8
    alpha = 0

    """load data"""

    if dataset_name == 'target':
        data_loader = get_data_loader(batch_size, normalize=False, target=True, train=False, transforms=None)
        tag = False
    else:
        data_loader = get_data_loader(batch_size, normalize=False, target=False, train=False, transforms=None)
        tag = True

    """ test """

    my_net = torch.load(os.path.join(
        './models/', '{0}/model_epoch_best.pth'.format(model_name)
    ))
    my_net = my_net.eval()
    #s = dict(my_net.named_modules())
    #print(dict(my_net.named_modules()))
    if cuda:
        my_net = my_net.cuda()

    len_dataloader = len(data_loader)
    data_target_iter = iter(data_loader)

    i = 0
    n_total = 0
    n_correct = 0
    layer_outputs = []  # 用于保存层输出的列表
    c_labels = []
    d_labels = []

    while i < len_dataloader:

        # test model using target data
        data_target = next(data_target_iter)
        t_img, t_label = data_target
        #t_img, t_label, domain_label = data_target

        batch_size = len(t_label)

        if cuda:
            t_img = t_img.cuda()
            t_label = t_label.cuda()

        t_img = t_img.unsqueeze(1)

        def hook(module, input, output):
            layer_outputs.append(output.detach().cpu().numpy())
            c_labels.append(t_label.cpu().numpy())
            #d_labels.append(domain_label.cpu().numpy())

        # 注册hook到选择的层
        layer = dict(my_net.named_modules())['class_classifier.c_fc3']
        hook_handle = layer.register_forward_hook(hook)

        # 判断是否是baseline模型
        class_output, domain_output = my_net(input_data=t_img, alpha=alpha)

        pred = class_output.data.max(1, keepdim=True)[1]
        n_correct += pred.eq(t_label.data.view_as(pred)).cpu().sum()
        n_total += batch_size

        if hook_handle is not None:
            # 取消hook
            hook_handle.remove()

        i += 1

    accu = n_correct.data.numpy() * 1.0 / n_total

    # 将层输出保存为.npy文件
    layer_outputs_np = np.concatenate(layer_outputs, axis=0)
    labels_outputs_np = np.concatenate(c_labels, axis=0)
    #d_outputs = np.concatenate(d_labels, axis=0)
    output_file = 'kat_outputt'
    np.save(output_file+'.npy', layer_outputs_np)
    np.save(output_file+'_c_label.npy', labels_outputs_np)
    #np.save(output_file + '_d_label.npy', d_outputs)
    print(f"Layer output saved to {output_file}.")

    return accu


accu_s = test('target', 'kat_1_3')
print(accu_s)
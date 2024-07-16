import os
import torch.backends.cudnn as cudnn
import torch.utils.data
from torchvision import transforms
from data_loader import get_data_loader
from torchvision import datasets


def test(dataset_name, model_name):
    assert dataset_name in ['source', 'target']

    cuda = True
    cudnn.benchmark = True
    batch_size = 8
    image_size = 28
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
        './models/', '{0}/model_epoch_current.pth'.format(model_name)
    ))
    my_net = my_net.eval()

    if cuda:
        my_net = my_net.cuda()

    len_dataloader = len(data_loader)
    data_target_iter = iter(data_loader)

    i = 0
    n_total = 0
    n_correct = 0

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
            # 判断是否是baseline模型
        class_output, domain_output = my_net(input_data=t_img, alpha=alpha)

        pred = class_output.data.max(1, keepdim=True)[1]
        n_correct += pred.eq(t_label.data.view_as(pred)).cpu().sum()
        n_total += batch_size

        i += 1

    accu = n_correct.data.numpy() * 1.0 / n_total

    return accu

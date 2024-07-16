import random
import os
import sys
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import numpy as np
from data_loader import get_data_loader
from model import CNNModel
from test import test
from torch.optim.lr_scheduler import ExponentialLR

model_name = 'kat_1_4'
cuda = True
cudnn.benchmark = True
lr = 1e-3
batch_size = 16
n_epoch = 50

manual_seed = random.randint(1, 10000)
random.seed(manual_seed)
torch.manual_seed(manual_seed)
path = './models/' + model_name
if not os.path.exists(path):
    os.makedirs(path)
    print(f"Directory {path} created.")
else:
    print(f"Directory {path} already exists.")

# load data

print('loading data...')
s_loader = get_data_loader(batch_size, normalize=False, target=False, train=True, transforms=None)
t_loader = get_data_loader(batch_size, normalize=False, target=True, train=True, transforms=None)

# load model

my_net = CNNModel()

# setup optimizer

optimizer = optim.Adam(my_net.parameters(), lr=lr)
#scheduler = ExponentialLR(optimizer, gamma=0.95)

loss_class = torch.nn.NLLLoss()
loss_domain = torch.nn.NLLLoss()

if cuda:
    my_net = my_net.cuda()
    loss_class = loss_class.cuda()
    loss_domain = loss_domain.cuda()

for p in my_net.parameters():
    p.requires_grad = True

# training
best_accu_t = 0.0
soruce_accs = []
target_accs = []
for epoch in range(n_epoch):

    len_dataloader = min(len(s_loader), len(t_loader))
    data_source_iter = iter(s_loader)
    data_target_iter = iter(t_loader)

    for i in range(len_dataloader):

        p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # training model using source data
        data_source = next(data_source_iter)
        s_img, s_label = data_source
        #s_img, s_label, domain_label = data_source

        my_net.zero_grad()
        batch_size = len(s_label)

        domain_label = torch.zeros(batch_size).long()

        if cuda:
            s_img = s_img.cuda()
            s_label = s_label.cuda()
            domain_label = domain_label.cuda()
        # 在维度1的位置增加一个新维度
        s_img = s_img.unsqueeze(1)  # 新尺寸是 [8, 1, 40, 40]

        class_output, domain_output = my_net(input_data=s_img, alpha=alpha)
        err_s_label = loss_class(class_output, s_label)
        err_s_domain = loss_domain(domain_output, domain_label)

        # training model using target data
        data_target = next(data_target_iter)
        t_img, _ = data_target
        #t_img, _, domain_label = data_target

        batch_size = len(t_img)

        domain_label = torch.ones(batch_size).long()

        if cuda:
            t_img = t_img.cuda()
            domain_label = domain_label.cuda()

        t_img = t_img.unsqueeze(1)

        _, domain_output = my_net(input_data=t_img, alpha=alpha)
        err_t_domain = loss_domain(domain_output, domain_label)
        err = err_t_domain + err_s_domain + err_s_label
        err.backward()
        optimizer.step()
        #scheduler.step()

        sys.stdout.write('\r epoch: %d, [iter: %d / all %d], err_s_label: %f, err_s_domain: %f, err_t_domain: %f' \
              % (epoch, i + 1, len_dataloader, err_s_label.data.cpu().numpy(),
                 err_s_domain.data.cpu().numpy(), err_t_domain.data.cpu().item()))
        sys.stdout.flush()
        torch.save(my_net, './models/{0}/model_epoch_current.pth'.format(model_name))

    print('\n')
    accu_s = test('source', model_name)
    print('Accuracy of the %s dataset: %f' % ('source', accu_s))
    accu_t = test('target', model_name)
    print('Accuracy of the %s dataset: %f\n' % ('target', accu_t))
    soruce_accs.append(accu_s)
    target_accs.append(accu_t)
    if accu_t > best_accu_t:
        best_accu_s = accu_s
        best_accu_t = accu_t
        torch.save(my_net, './models/{0}/model_epoch_best.pth'.format(model_name))
np.save('./models/{0}/t_accs.npy'.format(model_name), target_accs)
np.save('./models/{0}/s_accs.npy'.format(model_name), soruce_accs)


print('============ Summary ============= \n')
print('Accuracy of the %s dataset: %f' % ('source', best_accu_s))
print('Accuracy of the %s dataset: %f' % ('target', best_accu_t))
print('Corresponding model was save in ./models/' + model_name + '/model_epoch_best.pth')
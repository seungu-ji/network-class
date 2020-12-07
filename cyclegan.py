#필요한 라이브러리 import 
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import random
import numpy as np
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from abc import ABC, abstractmethod
from torch.nn import init
from matplotlib import pyplot as plt
import time
import cv2

#Cycle Gan 모델 선언
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)
    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]
        return nn.Sequential(*conv_block)
    def forward(self, x):
        out = x + self.conv_block(x)  # add skip connections
        return out
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]
        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]
        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]
        self.model = nn.Sequential(*model)
    def forward(self, input):
        return self.model(input)
# model 가중치 초기화 함수
def init_func(m):  # define the initialization function
        init_type='normal'
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, 0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
# 가중치 파일에서 필요없는 가중치 제거 및 몇몇 가중치 복제하는 함수
def patch_instance_norm_state_dict(state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)
        return state_dict
# input 이미지를  input 사이즈에 맞게 조절하고 tensor로 변환하는 함수
def get_transform( params=None, grayscale=False, method=Image.BICUBIC, convert=True):
    transform_list = []
    '''if grayscale:
        transform_list.append(transforms.Grayscale(1))'''
    osize = [256, 256]
    transform_list.append(transforms.Resize(osize, method))
    if params is None:
        transform_list.append(transforms.RandomCrop(256))
    else:
        transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], 256)))
    if params is None:
        pass
        #transform_list.append(transforms.RandomHorizontalFlip())
        '''elif params['flip']:
            transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))'''
    #여기 위에 없애기
    transform_list += [transforms.ToTensor()]
    transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)
# 텐서를 이미지(numpy.ndarray)로 바꿔주는 함수
def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.
    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)




def cyclegan_predict(img_path):
# 모델 선언
    norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    generator=ResnetGenerator(input_nc=3,output_nc=3,ngf=64, norm_layer=norm_layer, use_dropout=False, n_blocks=9, padding_type='reflect')
    #모델에 가중치 로드
    state=torch.load('./weight/style_vangogh.pth')
    if hasattr(state, '_metadata'):
      del state._metadata
    # patch InstanceNorm checkpoints prior to 0.4
    for key in list(state.keys()):  # need to copy keys here because we mutate in loop
        patch_instance_norm_state_dict(state, generator, key.split('.'))
    generator.load_state_dict(state)
    #gpu를  사용하기 위한 코드
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator.to(device)
    #이미지를 변환한 후 모델에 적용시킨 코드
    transform=get_transform()



    # 이미지 사이즈 확인
    size_img = cv2.imread(img_path)
    print(size_img.shape)
    row, col = size_img.shape[:2]
    print(row, col)

    # **중요**




    A_img = Image.open(img_path).convert('RGB')
    print(A_img)


    A = transform(A_img)
    # print(A)

    real = A.to(device)
    start_time = time.time()
    fake = generator(real.unsqueeze(0))
    end_time = time.time()
    print("WorkingTime: {} sec".format(end_time-start_time))
    #tensor를 이미지로 변환
    img=tensor2im(fake)
    #만약 사진이 좀 이상하다 하면은 아래코드주석치고 하번 더실행
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # plt 시각화
    # plt.imshow(img, interpolation='nearest')
    # plt.show()


    dst = cv2.resize(img, dsize=(col, row), interpolation=cv2.INTER_AREA)
    print(dst.shape)
    # cv2.imwrite('img2', dst)
    
    binary_cv = cv2.imencode('.PNG', dst)[1].tobytes()

    return binary_cv
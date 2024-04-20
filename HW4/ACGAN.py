#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as utils
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import time
from torch.utils.data import Subset
import torchvision.models as models
import torch.nn.functional as F
from scipy import linalg
import pandas as pd


# In[2]:


import os
if not os.path.exists('Results'):
   os.makedirs('Results')
if not os.path.exists('Models'):
   os.makedirs('Models')


# In[3]:


torch.manual_seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device:',device)
LEARNING_RATE = 5e-5
BATCH_SIZE = 64
IMAGE_SIZE = 64
CHANNELS_IMG = 3
NOISE_DIM = 100
NUM_EPOCHS = 40
FEATURES_DISC = 64
FEATURES_GEN = 64
NUM_CLASSES = 10
EMBED_SIZE = 100
criticItr = 5


# In[4]:


dataset = datasets.CIFAR10(root="./dataset/CIFAR10data", download=True,
                           transform=transforms.Compose([
                               transforms.Resize(64),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                         shuffle=True, num_workers=2)


# In[5]:


real_batch = next(iter(dataloader))

plt.figure(figsize=(10, 10))
plt.axis("off")
plt.title("Training Images")

image_grid = utils.make_grid(real_batch[0].to(device)[:BATCH_SIZE], padding=2, normalize=True).cpu()
image_grid = np.transpose(image_grid, (1, 2, 0))
plt.imshow(image_grid)
plt.show()


# In[6]:


class Generator(nn.Module):
    def __init__(self, num_classes, img_size,embed_size,channels_noise, channels_img, features_g):
        super(Generator, self).__init__()
        self.img_size = img_size
        self.net = nn.Sequential(
            nn.ConvTranspose2d(NOISE_DIM+embed_size,features_g* 16, 4, 1, 0, bias = False),
            self.Gnet(features_g * 16, features_g * 8, 4, 2, 1),
            self.Gnet(features_g * 8, features_g * 4, 4, 2, 1),
            self.Gnet(features_g * 4, features_g * 2, 4, 2, 1),
            nn.ConvTranspose2d(
                features_g * 2, channels_img, kernel_size=4, stride=2, padding=1
            ),
            nn.Tanh(),
        )

        self.embed = nn.Embedding(num_classes, embed_size)

    def Gnet(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels,momentum=0.9),
            nn.ReLU(),
        )

    def forward(self, x, labels):
        embedding = self.embed(labels).unsqueeze(2).unsqueeze(3)
        x = torch.cat([x, embedding], dim=1)
        return self.net(x)


# In[7]:


class Discriminator(nn.Module):
    def __init__(self, num_classes, img_size,channels_img, features_d):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(
                channels_img, features_d, kernel_size=4, stride=2, padding=1
            ),
            nn.LeakyReLU(0.2),
            self.Dnet(features_d, features_d * 2, 4, 2, 1),
            self.Dnet(features_d * 2, features_d * 4, 4, 2, 1),
            self.Dnet(features_d * 4, features_d * 8, 4, 2, 1),
        )
        self.validity_layer = nn.Sequential(
            nn.Conv2d(512,1,4,1,0,bias = False),
                               nn.Sigmoid())

        self.labels = nn.Sequential(
            nn.Conv2d(512, 11, 4, 1, 0, bias = False),
            nn.LogSoftmax(dim = 1)
                                        )
        self.embed = nn.Embedding(num_classes, img_size*img_size)



    def Dnet(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.LeakyReLU(0.2),
        )



    def forward(self, x):
        x = self.disc(x)
        valid = self.validity_layer(x).view(-1)
        label = self.labels(x).view(-1,11)
        return valid,label


# In[8]:


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


# In[9]:


gen = Generator(NUM_CLASSES, IMAGE_SIZE,EMBED_SIZE,NOISE_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
disc = Discriminator(NUM_CLASSES, IMAGE_SIZE,CHANNELS_IMG, FEATURES_DISC).to(device)
initialize_weights(gen)
initialize_weights(disc)


# In[10]:


gen.train() , disc.train()


# In[11]:


class InceptionV3(nn.Module):
    """Pretrained InceptionV3 network returning feature maps"""
    DEFAULT_BLOCK_INDEX = 3

    BLOCK_INDEX_BY_DIM = {
        64: 0,
        192: 1,
        768: 2,
        2048: 3
    }

    def __init__(self,
                 output_blocks=[DEFAULT_BLOCK_INDEX],
                 resize_input=True,
                 normalize_input=True,
                 requires_grad=False):

        super(InceptionV3, self).__init__()

        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)

        assert self.last_needed_block <= 3,             'Last possible output block index is 3'

        self.blocks = nn.ModuleList()


        inception = models.inception_v3(pretrained=True)

        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        self.blocks.append(nn.Sequential(*block0))

        if self.last_needed_block >= 1:
            block1 = [
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2)
            ]
            self.blocks.append(nn.Sequential(*block1))

        if self.last_needed_block >= 2:
            block2 = [
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))

        if self.last_needed_block >= 3:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
            ]
            self.blocks.append(nn.Sequential(*block3))

        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, inp):
        outp = []
        x = inp

        if self.resize_input:
            x = F.interpolate(x,
                              size=(299, 299),
                              mode='bilinear',
                              align_corners=False)

        if self.normalize_input:
            x = 2 * x - 1

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)

            if idx == self.last_needed_block:
                break

        return outp

block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
model = InceptionV3([block_idx])
model=model.cuda()

def calculate_activation_statistics(images,model,batch_size=128, dims=2048,
                    cuda=False):
    model.eval()
    act=np.empty((len(images), dims))

    if cuda:
        batch=images.cuda()
    else:
        batch=images
    pred = model(batch)[0]

    if pred.size(2) != 1 or pred.size(3) != 1:
        pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

    act= pred.cpu().data.numpy().reshape(pred.size(0), -1)

    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    from scipy.linalg import sqrtm
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, 'Mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, 'Covariance matrices have different dimensions'

    sigma1 += np.eye(sigma1.shape[0]) * eps
    sigma2 += np.eye(sigma2.shape[0]) * eps

    mean_diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)

    # Check for numerical issues
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError('Imaginary components detected in covmean')
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    # The Frechet distance formula
    return (mean_diff.dot(mean_diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

def calculate_fretchet(images_real,images_fake,model):
    mu_1,std_1=calculate_activation_statistics(images_real,model,cuda=True)
    mu_2,std_2=calculate_activation_statistics(images_fake,model,cuda=True)

    """get fretched distance"""
    fid_value = calculate_frechet_distance(mu_1, std_1, mu_2, std_2)
    return fid_value


# In[12]:


fixed_noise = torch.randn(32, NOISE_DIM, 1, 1).to(device)
step = 0


# In[13]:


disOptim = optim.Adam(disc.parameters(),LEARNING_RATE, betas = (0.5,0.999))
genOptim = optim.Adam(gen.parameters(), LEARNING_RATE, betas = (0.5,0.999))
criterion = nn.BCELoss()
counterList = []
counter = 0
genLossList = []
disLossList = []
img_list = []


# In[14]:


GenLoss = []
DiscLoss = []
img_list = []
iters = 0
FID_list = []

print("Started Training")
time_start = time.time()
for epoch in range(NUM_EPOCHS):
    epoch = epoch+1


    for batch_idx, (realImg,labels) in enumerate(dataloader):
        real = realImg.to(device)
        labels = labels.to(device)
        miniBatch = real.shape[0]
        fakeClassLabel =  10*torch.ones((miniBatch,),dtype = torch.long).to(device)
        disc.zero_grad()

        r_label =  torch.full((miniBatch,), 1.0, dtype=torch.float).to(device)
        disc_real, disc_label = disc(real)
        loss_disc_realVal = criterion(disc_real, r_label)
        discRealLabelLoss = F.nll_loss(disc_label, labels)
        realLoss_disc = loss_disc_realVal + discRealLabelLoss
        realLoss_disc.backward()

        sampleLabel = torch.randint(0, 10, (miniBatch,), dtype = torch.long).to(device)
        noise = torch.randn(miniBatch, NOISE_DIM, 1, 1).to(device)
        fake = gen(noise,sampleLabel)
        disc_fake, disc_fLabel = disc(fake.detach())
        r_label.fill_(0.0)
        loss_disc_fakelVal = criterion(disc_fake, r_label)
        discFakeLabelLoss = F.nll_loss(disc_fLabel, fakeClassLabel)
        fakeLoss_disc = loss_disc_fakelVal + discFakeLabelLoss
        fakeLoss_disc.backward()
        loss_disc = (realLoss_disc+fakeLoss_disc)/2
        disOptim.step()

        gen.zero_grad()
        noise = torch.randn(miniBatch, NOISE_DIM, 1, 1).to(device)
        sampleLabel = torch.randint(0,10,(miniBatch,),dtype = torch.long).to(device)
        r_label.fill_(1.0)
        fake = gen(noise, sampleLabel)
        genOutput, genOutputLabel = disc(fake)
        loss_gen_r = criterion(genOutput, r_label)
        genLabelLoss = F.nll_loss(genOutputLabel, sampleLabel)
        loss_gen = loss_gen_r + genLabelLoss
        loss_gen.backward()
        genOptim.step()
        GenLoss.append(loss_gen.detach())
        DiscLoss.append(loss_disc.detach())

        if (iters % 500 == 0) or ((epoch == NUM_EPOCHS) and (batch_idx == len(dataloader)-1)):
            with torch.no_grad():
                fake = gen(noise,sampleLabel).detach()
            img_list.append(utils.make_grid(fake, padding=2, normalize=True))

        iters += 1

    #Cal. FID
    fretchet_dist=calculate_fretchet(real,fake,model)
    FID_list.append(fretchet_dist)
    if epoch % 5  == 0:
               print(f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(dataloader)}                   DLoss: {loss_disc:.3f}, GLoss: {loss_gen:.3f}, FID Score: {fretchet_dist:.3f} ")


    #Saving Generated image/epoch
    if not os.path.exists('Results/ACGAN_FAKE'):
       os.makedirs('Results/ACGAN_FAKE')
    if not os.path.exists('Results/ACGAN_REAL'):
       os.makedirs('Results/ACGAN_REAL')
    images = gen(noise,sampleLabel).detach()
    utils.save_image(images.detach(),'Results/ACGAN_FAKE/ACGAN_epoch_%03d.png' % (epoch), normalize = True)

    #Saving Real image/epoch
    utils.save_image(real.detach(),'Results/ACGAN_REAL/ACGAN_epoch_%03d.png' % (epoch), normalize = True)


time_end = time.time()
print('\n\n elapsed time：%.2f s.'%(time_end-time_start))


# In[16]:


GenLoss


# In[17]:


DiscLoss


# In[20]:


# Convert lists of tensors to NumPy arrays
GenLoss_cpu = [loss.cpu().numpy() for loss in GenLoss]
DiscLoss_cpu = [loss.cpu().numpy() for loss in DiscLoss]

# Ensure the directory exists before saving
if not os.path.exists('Results/Loss_Data'):
    os.makedirs('Results/Loss_Data')

# Save the numpy arrays
np.save('Results/Loss_Data/ACGAN_GLoss', GenLoss_cpu)
np.save('Results/Loss_Data/ACGAN_DLoss', DiscLoss_cpu)


# In[21]:


GenLoss = np.load('Results/Loss_Data/ACGAN_GLoss.npy')
DiscLoss = np.load('Results/Loss_Data/ACGAN_DLoss.npy')


# In[22]:


plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")

plt.plot(GenLoss, label="Generator")
plt.plot(DiscLoss, label="Discriminator")

plt.xlabel("Iterations")
plt.ylabel("Loss")

plt.legend()

plt.savefig('Results/LOSS.png', format='png', dpi=100, bbox_inches='tight')

plt.show()


# In[26]:



real_batch = next(iter(dataloader))

plt.figure(figsize=(15, 15))


plt.subplot(1, 2, 1)
plt.axis("off")
plt.title("Real Images")
real_images_grid = utils.make_grid(real_batch[0][:10], nrow=5, padding=5, normalize=True)
real_images_np = real_images_grid.cpu().numpy()
real_images_np = np.transpose(real_images_np, (1, 2, 0))
plt.imshow(real_images_np)

plt.subplot(1, 2, 2)
plt.axis("off")
plt.title("Fake Images")
fake_images_np = img_list[-1].cpu().numpy()
fake_images_np = np.transpose(fake_images_np, (1, 2, 0))
plt.imshow(fake_images_np)

plt.savefig('Results/ACGAN_BEST_FAKE.png', format='png', dpi=100, bbox_inches='tight')

# Show the plot
plt.show()


# In[27]:


plt.figure(figsize=(10,5))
plt.title("FID Scores for ACGAN")
plt.plot(FID_list,label="ACGAN")
plt.xlabel("Epochs")
plt.ylabel("FID")
plt.legend()
plt.savefig('Results/FID.png',
            format='png',
            bbox_inches='tight')
plt.show()


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[28]:


if not os.path.exists('Results/FID_score'):
       os.makedirs('Results/FID_score')
np.save('Results/FID_score/ACGAN_FID', FID_list)


# In[29]:


print(np.mean(GenLoss),np.mean(DiscLoss),np.min(GenLoss),np.min(DiscLoss),GenLoss[-1],DiscLoss[-1])
print(np.min(FID_list), np.max(FID_list), np.mean(FID_list), FID_list[-5])


# In[30]:


import os
from google.colab import files
import shutil

# Zip the 'Results' folder
shutil.make_archive('Results', 'zip', 'Results')

# Download the zipped folder
files.download('Results.zip')


# In[ ]:





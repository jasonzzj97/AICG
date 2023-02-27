# Setup environment
import numpy as np
import pandas as pd
import mdtraj as md
import glob
import os

import torch.nn as nn
import torch

from src.feature import (GeometryFeature, GeometryStatistics,
                           MoleculeDataset, LinearLayer)
from src.network import (CGnet, HarmonicLayer,HarmonicLayer_L, ForceLoss, ZscoreLayer,RepulsionLayer_L,
		LJ_L, FENE_bond_L, REB_angle_L, AttractionLayer_L,HarmonicLayer_L_angle,HarmonicLayer_L_k, LJ_namd,
                           lipschitz_projection, dataset_loss, Simulation, CGnet_FP, CGnet_LP, dataset_loss_FP)

from src.molecule import CGMolecule
from torch.utils.data import DataLoader, RandomSampler
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.model_selection import train_test_split
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

# We specify the CPU as the training/simulating device here.
# If you have machine  with a GPU, you can use the GPU for
# accelerated training/simulation by specifying 
# device = torch.device('cuda')
device = torch.device('cpu')
print(device)


# Load data
def csv2xyz(data_coors, N, start, t='coor'):
  total_time = len(data_coors)/N
  total_frame = []
  for i in range(start, start+int(total_time)):
    curr_coor = data_coors.loc[lambda df: data_coors['Time (ps)'] == i+1, :]
    if t == 'coor':
      curr_x = np.array(curr_coor['coor x'])
      curr_y = np.array(curr_coor['coor y'])
      curr_z = np.array(curr_coor['coor z'])
    else:
      curr_x = np.array(curr_coor['force x'])
      curr_y = np.array(curr_coor['force y'])
      curr_z = np.array(curr_coor['force z'])
    curr_frame = [[curr_x[i], curr_y[i], curr_z[i]] for i in range(len(curr_x))]
    total_frame.append(curr_frame)
  return np.array(total_frame)

c_name = glob.glob('data/coors/pd*_coors.csv')
f_name = glob.glob('data/forces/pd*_forces.csv')
#c_name = glob.glob('data/coors/pd8_cg_coors.csv')
#f_name = glob.glob('data/forces/pd8_cg_forces.csv')


for n, name in enumerate(c_name):
  data = pd.read_csv(name, delimiter='\t')
  cg_xyz = csv2xyz(data, 15, 0, 'coor')
  if n == 0:
    coords = cg_xyz
  else:
    coords = np.vstack((coords, cg_xyz))
  if name[-14] == '2':
    coords = np.delete(coords, slice(13000, 14000), 0)

for n, name in enumerate(f_name):
  data = pd.read_csv(name, delimiter='\t')
  cg_xyz = csv2xyz(data, 15, 0, 'force')
  if n == 0:
    forces = cg_xyz
  else:
    forces = np.vstack((forces, cg_xyz))
  if name[-14] == '2':
    forces = np.delete(forces, slice(13000, 14000), 0)


#np.random.shuffle(coords)
#np.random.shuffle(forces)

coords = coords.astype('float32')
forces = forces.astype('float32')
print("Coordinates size: {}".format(coords.shape))
print("Force: {}".format(forces.shape))


#cg_topo = md.load_psf('data/cg_3GHG_autopsf.psf')
#_, bonds = cg_topo.to_dataframe()
#bonds=bonds[:,:2]


# Network

#custom_bond = [(5,7), (5,14),(12,14),(12,13),(8,13),(8,9),(2,9),(2,3),(3,4),(4,10),(0,11),(11,12),(2,6),(1,6)]
#custom_angle = [(7,5,14), (5,14,12),(14,12,13),(12,13,8),(13,8,9),(8,9,2),(9,2,3),(2,3,4),(3,4,10),(0,11,12),(11,12,14), (11,12,13),(1,6,2),(6,2,9),(6,2,3)]
custom_bond = [(0,11), (1,6),(2,3),(2,6),(2,9),(3,4),(4,10),(5,7),(5,14),(8,9),(8,13),(11,12),(12,13),(12,14)]
custom_angle = [(0,11,12),(1,6,2),(2,9,8),(2,3,4),(3,4,10),(3,2,9),(3,2,6),(5,14,12),(6,2,9),(7,5,14),(8,13,12),(9,8,13),(11,12,14),(11,12,13),(13,12,14)]
custom_dihedral = [(0,11,12,14),(0,11,12,13),(1,6,2,3),(1,6,2,9),(11,12,13,8),(11,12,14,5),(6,2,3,4),(6,2,9,8), (7,5,14,12),(5,14,12,13),(14,12,13,8), (12,13,8,9),(13,8,9,2),(8,9,2,3),(9,2,3,4),(2,3,4,10)]

## energy unit: kj/mol
stats = GeometryStatistics(coords, backbone_inds='all',custom_feature_tuples = custom_angle + custom_dihedral,
                           get_all_distances=True,
                           get_backbone_angles=False,
                           get_backbone_dihedrals=False, temperature=310.0)

bond_list, bond_keys = stats.get_prior_statistics(features=custom_bond, as_list=True)
bond_indices = stats.return_indices(features=custom_bond)

angle_list, angle_keys = stats.get_prior_statistics(features='Angles', as_list=True)
angle_indices = stats.return_indices('Angles')
angle_scaling = 1
for i in range(len(angle_list)):
    angle_list[i]['k']*=angle_scaling

print("We have {} backbone beads, {} bonds, and {} angles.".format(
                        coords.shape[1], len(bond_list), len(angle_list)))
print("Bonds: ")
for key, stat in zip(bond_keys, bond_list):
    print("{} : {}".format(key, stat))
print("Angles: ")
for key, stat in zip(angle_keys, angle_list):
    print("{} : {}".format(key, stat))

dihedral_list, dihedral_keys = stats.get_prior_statistics(features='Dihedral_angles', as_list=True)
dihedral_indices = stats.return_indices('Dihedral_angles')
print("We have {} dihedrals.".format(len(dihedral_list)))
print("Dihedrals: ")
for key, stat in zip(dihedral_keys, dihedral_list):
    print("{} : {}".format(key, stat))

all_stats, what = stats.get_prior_statistics(as_list=True)
num_feats = len(all_stats)
zscores, _ = stats.get_zscore_array()
print("We have {} statistics for {} features.".format(zscores.shape[0], zscores.shape[1]))


def read_para_n0(filename):
    f = open(filename, "r")
    Lines = f.readlines()
    bond=[]
    angle=[]
    dihedral=[]
    lj = []
    for i in range(9,23):
        bond.append(float(Lines[i].split()[2]))
    for i in range(33,48):
        angle.append(float(Lines[i].split()[3]))
    for i in range(58,74):
        dihedral.append(float(Lines[i].split()[4]))
    for i in range(84,99):
        lj.append([float(Lines[i].split()[2]),float(Lines[i].split()[3])])
    return bond,angle,dihedral,lj

bond=[]
angle=[]
dihedral=[]
lj=[]
para_path= 'data/iteration10.par'
b, a, d, l = read_para_n0(para_path)
bond.append(b)
angle.append(a)
dihedral.append(d)
lj.append(l)
bond = np.array(bond)
angle = np.array(angle)
dihedral = np.array(dihedral)
lj = np.array(lj)

#pair_indices = [i for i in range(105)]
pair_indices = [i for i in [j for j in range(105)] if i not in bond_indices]
pair_list = [{'eps': torch.tensor(lj[0,i,0]*4.184), 'rmin': torch.tensor(lj[0,i,1]/10)} for i in range(15)]


for i in range(len(bond_list)):
    bond_list[i]['k'] = torch.tensor(bond[0][i]*836.8)
for i in range(len(angle_list)):
    angle_list[i]['k'] = torch.tensor(angle[0][i]*8.368)
for i in range(len(dihedral_list)):
    dihedral_list[i]['k'] = torch.tensor(dihedral[0][i]*8.368)


# Hyperparameters

n_layers = 5 
n_nodes = 350 
activation = nn.ReLU()
batch_size = 512 
learning_rate = 1e-4
#rate_decay = 0.5 
#lr_milestone = [3000, 6000, 9000]
lipschitz_strength = 4.0

now = datetime.now()
stamp = now.strftime("%Y%m%d-%H%M%S")

num_epochs = 300

verbose = True
save_model = True
batch_freq = 100
epoch_freq = 1

restart_freq = 50



# Start by scaling according to mean and standard deviation
layers = [ZscoreLayer(zscores)]

# The first hidden layer goes from number of features to 160
num_feat = len(all_stats)
layers += LinearLayer(num_feat, n_nodes, activation=nn.Tanh())

# The inner hidden layers stay the same size
for _ in range(n_layers - 2):
    layers += LinearLayer(n_nodes, n_nodes, activation=nn.ReLU())

layers += LinearLayer(n_nodes, n_nodes, activation=nn.ReLU(), dropout=0.5)
# The last layer produces a single value
layers += LinearLayer(n_nodes, 1, activation=None)

# Construct prior energy layers
priors  = [HarmonicLayer_L_k(bond_indices, bond_list)]
priors += [HarmonicLayer_L_k(angle_indices, angle_list)]
priors += [LJ_namd(pair_indices, pair_list)]
priors += [HarmonicLayer_L_k(dihedral_indices, dihedral_list)]

feature_layer = GeometryFeature(feature_tuples=stats.feature_tuples, device=device)


#directory = 'logs/20210711-181456/'
directory = './logs/' + stamp + '/' # to save model
os.makedirs(directory, exist_ok=True)
print(directory)

#fib_net = CGnet(layers, ForceLoss(), feature=feature_layer, priors=None).to(device)
fib_net = CGnet_LP(layers, ForceLoss(), feature=feature_layer, priors=priors).to(device)
#fib_net = CGnet_FP(layers, ForceLoss(), feature=feature_layer, priors=priors, bond_in_feature = bond_in_feature, resting_para = zscores[0].to(device), device = device).to(device)
#fib_net = torch.load(directory + 'fib_net.pt', map_location=device)

print(fib_net)
fib_net.mount(device)

for name, param in fib_net.named_parameters():
	    if param.requires_grad:
			        print(name, param.data)

# Training
rs = 36

c_train, c_test, f_train, f_test = train_test_split(coords, forces, test_size=0.05, random_state=rs)

fib_data_train = MoleculeDataset(c_train, f_train, device=device)
fib_data_test = MoleculeDataset(c_test, f_test, device=device)
print("Training set length: {}".format(len(fib_data_train)))
print("Test set length: {}".format(len(fib_data_test)))

#with open(directory + 'training_log.txt', 'w') as f:
#	print(device, file = f)
#	print("Training set length: {}".format(len(fib_data_train)), file = f)
#	print("Test set length: {}".format(len(fib_data_test)), file = f)
#	print(fib_net, file = f)


# Training tools

trainloader = DataLoader(fib_data_train, sampler=RandomSampler(fib_data_train),
                         batch_size=batch_size)

testloader = DataLoader(fib_data_test, sampler=RandomSampler(fib_data_test),
                         batch_size=batch_size)

optimizer = torch.optim.Adam(fib_net.parameters(),
                             lr=learning_rate)
def regularizer(model, strength=lipschitz_strength):
        lipschitz_projection(model, strength=strength)

#scheduler = MultiStepLR(optimizer,milestones=lr_milestone,
#                        gamma=rate_decay)
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-6, max_lr=learning_rate,step_size_up=100,mode="triangular2", cycle_momentum=False)

tb = SummaryWriter(log_dir = directory)

for epoch in range(1, num_epochs+1):
    train_loss = dataset_loss(fib_net, trainloader,optimizer,verbose_interval=batch_freq)
    test_loss = dataset_loss(fib_net, testloader, train_mode=False)

    if verbose:
        if epoch % epoch_freq == 0:
            print(
            "Epoch: {} | Train loss: {:.2f} | Test loss: {:.2f}\n".format(epoch, train_loss, test_loss))
            tb.add_scalar('Training Total Loss', train_loss, epoch)
            tb.add_scalar('Testing Total Loss', test_loss, epoch)
            for name, param in fib_net.named_parameters():
                if param.requires_grad:
                    if name[0] == 'a':
                        tb.add_histogram(name, param, epoch)
                    else:
                        tb.add_histogram(name + '_sq', torch.square(param), epoch)
    if save_model:
        if epoch % restart_freq == 0:
            torch.save(fib_net,"{}/fib_net.pt".format(directory))

    scheduler.step()

tb.close()

# Author: Nick Charron
# Contributors: Brooke Husic, Dominik Lemm, Jiang Wang, Simon Olsson

import torch
import numpy as np
import torch.nn as nn

class _AbstractPriorLayer(nn.Module):
    """Abstract Layer for definition of priors, which only imposes the minimal
    functional constraints to enable model estimation and inference. 
    """
    def __init__(self):
        super(_AbstractPriorLayer, self).__init__()
        self.callback_indices = slice(None, None)

    def forward(self, x):
        """Forward method to compute the prior energy contribution.

        Notes
        -----
        This must be explicitly implemented in a child class that inherits from
        _AbstractPriorLayer(). The details of this method should encompass the
        mathematical steps to form each specific energy contribution to the
        potential energy.;
        """
        raise NotImplementedError(
            'forward() method must be overridden in \
            custom classes inheriting from _AbstractPriorLayer()'
                                  )

class _PriorLayer(_AbstractPriorLayer):
    """Layer for adding prior energy computations external to CGnet hidden
    output

    Parameters
    ----------
    callback_indices: list of int
        indices used to access a specified subset of outputs from the feature
        layer through a residual connection

    interaction_parameters : list of python dictionaries
        list of dictionaries that specify the constants characterizing
        interactions between beads. Each list element corresponds to a single
        interaction using a dictionary of parameters keyed to corresponding
        numerical values. The order of these dictionaries follows the same order
        as the callback indices specifying which outputs from the feature layer
        should pass through the prior. The strucutre of interaction_parameters
        is the following:

            [ {'parameter_1' : 1.24, 'parameter_2' : 2.21, ... },
              {'parameter_1' : 1.24, 'parameter_2' : 2.21, ... },
                                     .
                                     .
                                     .
              {'parameter_1' : 1.24, 'parameter_2' : 2.21, ... }]

        In this way, _PriorLayer may be subclassed to make arbitray prior
        layers based on arbitrary interactions between bead tuples.

    Attributes
    ----------
    interaction_parameters: list of dict
        each list element contains a dictionary of physical parameters that
        characterizxe the interaction of the associated beads. The order of
        this list proceeds in the same order as self.callback_indices
    callback_indices: list of int
        indices used to access a specified subset of outputs from the feature
        layer through a residual connection

    Examples
    --------
    To assemble the feat_dict input for a HarmonicLayer prior for bonds from an
    instance of a stats = GeometryStatistics():

    bonds_interactions, _ = stats.get_prior_statistics('Bonds', as_list=True)
    bonds_idx = stats.return_indices('Bonds')
    bond_layer = HarmonicLayer(bonds_idx, bonds_interactions)

    Notes
    -----
    callback_indices and interaction_parameters MUST share the same order for
    the prior layer to produce correct energies. Using
    GeometryStatistics.get_prior_statistics() with as_list=True together with
    GeometryStatistics.return_indices() will ensure this is True for the same
    list of features.

    The units of the interaction paramters for priors must correspond with the
    units of the input coordinates and force labels used to train the CGnet.

    """

    def __init__(self, callback_indices, interaction_parameters):
        super(_PriorLayer, self).__init__()
        # if len(callback_indices) != len(interaction_parameters):
        #     raise ValueError(
        #         "callback_indices and interaction_parameters must have the same length"
        #         )
        self.interaction_parameters = interaction_parameters
        self.callback_indices = callback_indices

class RepulsionLayer(_PriorLayer):
    """Layer for calculating pairwise repulsion energy prior. Pairwise repulsion
    energies are calculated using the following formula:

        U_repulsion_ij = (sigma_{ij} / r_{ij}) ^ exp_{ij}

    where U_repulsion_ij is the repulsion energy contribution from
    coarse grain beads i and j, sigma_ij is the excluded volume parameter
    between the pair (in units of distance), r_ij is the pairwise distance
    (in units of distance) between coarse grain beads i and j, and exp_ij
    is the repulsion exponenent (dimensionless) that characterizes the
    asymptotics of the interaction.

    Parameters
    ----------
    callback_indices: list of int
        indices used to access a specified subset of outputs from the feature
        layer through a residual connection

    interaction_parameters : list of python dictionaries
        list of dictionaries that specify the constants characterizing
        interactions between beads. Each list element corresponds to a single
        interaction using a dictionary of parameters keyed to corresponding
        numerical values. The order of these dictionaries follows the same order
        as the callback indices specifying which outputs from the feature layer
        should pass through the prior. The parameters for RepulsionLayer
        dictionaries are 'ex_vol', the excluded volume (in length units), and
        'exp', the (positive) exponent characterizing the repulsion strength
        decay with distance.

    Attributes
    ----------
    repulsion_parameters : torch.Tensor
        tensor of shape [2, num_interactions]. The first row contains the
        excluded volumes, the second row contains the exponents, and each
        column corresponds to a single interaction in the order determined
        by self.callback_indices

    Notes
    -----
    This prior energy should be used for longer molecules that may possess
    metastable states in which portions of the molecule that are separated by
    many CG beads in sequence may nonetheless adopt close physical proximities.
    Without this prior, it is possilbe for the CGnet to learn energies that do
    not respect proper physical pairwise repulsions. The interaction is modeled
    after the VDW interaction term from the classic Leonard Jones potential.

    References
    ----------
    Wang, J., Olsson, S., Wehmeyer, C., PÃ©rez, A., Charron, N. E.,
        de Fabritiis, G., NoÃ©, F., Clementi, C. (2019). Machine Learning
        of Coarse-Grained Molecular Dynamics Force Fields. ACS Central Science.
        https://doi.org/10.1021/acscentsci.8b00913
    """

    def __init__(self, callback_indices, interaction_parameters):
        super(RepulsionLayer, self).__init__(
            callback_indices, interaction_parameters)
        for param_dict in self.interaction_parameters:
            if (key in param_dict for key in ('ex_vol', 'exp')):
                pass
            else:
                raise KeyError(
                    'Missing or incorrect key for repulsion parameters'
                )
        repulsion_parameters = torch.tensor([])
        for param_dict in self.interaction_parameters:
            repulsion_parameters = torch.cat((
                repulsion_parameters,
                torch.tensor([[param_dict['ex_vol']],
                              [param_dict['exp']]])), dim=1)
        self.register_buffer('repulsion_parameters', repulsion_parameters)

    def forward(self, in_feat):
        """Calculates repulsion interaction contributions to energy

        Parameters
        ----------
        in_feat: torch.Tensor
            input features, such as pairwise distances, of size (n,k), for
            n examples and k features.
        Returns
        -------
        energy: torch.Tensor
            output energy of size (n,1) for n examples.
        """

        n = len(in_feat)
        energy = torch.sum((self.repulsion_parameters[0, :]/in_feat)
                           ** self.repulsion_parameters[1, :],
                           1).reshape(n, 1) / 2
        return energy


class RepulsionLayer_L(_PriorLayer):
    """Layer for calculating pairwise repulsion energy prior. Pairwise repulsion
    energies are calculated using the following formula:

        U_repulsion_ij = (sigma_{ij} / r_{ij}) ^ exp_{ij}

    where U_repulsion_ij is the repulsion energy contribution from
    coarse grain beads i and j, sigma_ij is the excluded volume parameter
    between the pair (in units of distance), r_ij is the pairwise distance
    (in units of distance) between coarse grain beads i and j, and exp_ij
    is the repulsion exponenent (dimensionless) that characterizes the
    asymptotics of the interaction.

    Parameters
    ----------
    callback_indices: list of int
        indices used to access a specified subset of outputs from the feature
        layer through a residual connection

    interaction_parameters : list of python dictionaries
        list of dictionaries that specify the constants characterizing
        interactions between beads. Each list element corresponds to a single
        interaction using a dictionary of parameters keyed to corresponding
        numerical values. The order of these dictionaries follows the same order
        as the callback indices specifying which outputs from the feature layer
        should pass through the prior. The parameters for RepulsionLayer
        dictionaries are 'ex_vol', the excluded volume (in length units), and
        'exp', the (positive) exponent characterizing the repulsion strength
        decay with distance.

    Attributes
    ----------
    repulsion_parameters : torch.Tensor
        tensor of shape [2, num_interactions]. The first row contains the
        excluded volumes, the second row contains the exponents, and each
        column corresponds to a single interaction in the order determined
        by self.callback_indices

    Notes
    -----
    This prior energy should be used for longer molecules that may possess
    metastable states in which portions of the molecule that are separated by
    many CG beads in sequence may nonetheless adopt close physical proximities.
    Without this prior, it is possilbe for the CGnet to learn energies that do
    not respect proper physical pairwise repulsions. The interaction is modeled
    after the VDW interaction term from the classic Leonard Jones potential.

    References
    ----------
    Wang, J., Olsson, S., Wehmeyer, C., PÃ©rez, A., Charron, N. E.,
        de Fabritiis, G., NoÃ©, F., Clementi, C. (2019). Machine Learning
        of Coarse-Grained Molecular Dynamics Force Fields. ACS Central Science.
        https://doi.org/10.1021/acscentsci.8b00913
    """

    def __init__(self, callback_indices, interaction_parameters):
        super(RepulsionLayer_L, self).__init__(
            callback_indices, interaction_parameters)
        for param_dict in self.interaction_parameters:
            if (key in param_dict for key in ('ex_vol', 'exp')):
                pass
            else:
                raise KeyError(
                    'Missing or incorrect key for repulsion parameters'
                )
                
        r_parameters = torch.tensor([], requires_grad=True)
        for param_dict in self.interaction_parameters:
            r_parameters = torch.cat((
                r_parameters,
                torch.sqrt(torch.tensor([[param_dict['ex_vol']],
                              [param_dict['exp']]], requires_grad=True))), dim=1)
        repulsion_parameters = nn.Parameter(r_parameters)
        
        self.register_parameter('repulsion_parameters', repulsion_parameters)

    def forward(self, in_feat):
        """Calculates repulsion interaction contributions to energy

        Parameters
        ----------
        in_feat: torch.Tensor
            input features, such as pairwise distances, of size (n,k), for
            n examples and k features.
        Returns
        -------
        energy: torch.Tensor
            output energy of size (n,1) for n examples.
        """

        n = len(in_feat)
        energy = torch.sum(((self.repulsion_parameters[0, :] ** 2)/in_feat)
                           ** (self.repulsion_parameters[1, :] ** 2),
                           1).reshape(n, 1)
        return energy

class AttractionLayer_L(_PriorLayer):

    def __init__(self, callback_indices, interaction_parameters):
        super(AttractionLayer_L, self).__init__(
            callback_indices, interaction_parameters)
        for param_dict in self.interaction_parameters:
            if (key in param_dict for key in ('ex_vol', 'exp')):
                pass
            else:
                raise KeyError(
                    'Missing or incorrect key for repulsion parameters'
                )
                
        a_parameters = torch.tensor([], requires_grad=True)
        for param_dict in self.interaction_parameters:
            a_parameters = torch.cat((
                a_parameters,
                torch.sqrt(torch.tensor([[param_dict['ex_vol']],
                              [param_dict['exp']]], requires_grad=True))), dim=1)
        attraction_parameters = nn.Parameter(a_parameters)
        
        self.register_parameter('attraction_parameters', attraction_parameters)

    def forward(self, in_feat):
        """Calculates repulsion interaction contributions to energy

        Parameters
        ----------
        in_feat: torch.Tensor
            input features, such as pairwise distances, of size (n,k), for
            n examples and k features.
        Returns
        -------
        energy: torch.Tensor
            output energy of size (n,1) for n examples.
        """

        n = len(in_feat)
        energy = -torch.sum(((self.attraction_parameters[0, :] ** 2)/in_feat)
                           ** (self.attraction_parameters[1, :] ** 2),
                           1).reshape(n, 1)
        return energy

class LJ(_PriorLayer):
    """Layer for calculating pairwise LJ energy prior.
    """

    def __init__(self, callback_indices, interaction_parameters):
        super(LJ, self).__init__(
            callback_indices, interaction_parameters)
        for param_dict in self.interaction_parameters:
            if (key in param_dict for key in ('c12', 'c6', 'a', 'b')):
                pass
            else:
                raise KeyError(
                    'Missing or incorrect key for repulsion parameters'
                )
                
        LJ_parameters = torch.tensor([])
        for param_dict in self.interaction_parameters:
            LJ_parameters = torch.cat((
                LJ_parameters,
                torch.tensor([[param_dict['c12']],
                              [param_dict['c6']],
                              [param_dict['a']],
                              [param_dict['b']]])), dim=1)
        
        self.register_buffer('LJ_parameters', LJ_parameters)

    def forward(self, in_feat):
        """
        Returns
        -------
        energy: torch.Tensor
            output energy of size (n,1) for n examples.
        """

        n = len(in_feat)
        energy = torch.sum((self.LJ_parameters[0, :]/in_feat)
                           ** self.LJ_parameters[2, :] - (self.LJ_parameters[1, :]/in_feat)
                           ** self.LJ_parameters[3, :], 1).reshape(n, 1)
        return energy


class LJ_L(_PriorLayer):
    """Layer for calculating pairwise LJ energy prior.
    """

    def __init__(self, callback_indices, interaction_parameters):
        super(LJ_L, self).__init__(
            callback_indices, interaction_parameters)
        for param_dict in self.interaction_parameters:
            if (key in param_dict for key in ('c12', 'c6', 'a', 'b')):
                pass
            else:
                raise KeyError(
                    'Missing or incorrect key for repulsion parameters'
                )
                
        lj_parameters = torch.tensor([], requires_grad=True)
        for param_dict in self.interaction_parameters:
            lj_parameters = torch.cat((
                lj_parameters,
                torch.sqrt(torch.tensor([[param_dict['c12']],
                              [param_dict['c6']],
                              [param_dict['a']],
                              [param_dict['b']]], requires_grad=True))), dim=1)
        LJ_parameters = nn.Parameter(lj_parameters)
        
        self.register_parameter('LJ_parameters', LJ_parameters)

    def forward(self, in_feat):
        """
        Returns
        -------
        energy: torch.Tensor
            output energy of size (n,1) for n examples.
        """

        n = len(in_feat)
        energy = torch.sum(((self.LJ_parameters[0, :] ** 2)/in_feat)
                           ** (self.LJ_parameters[2, :] ** 2) - ((self.LJ_parameters[1, :] ** 2)/in_feat)
                           ** (self.LJ_parameters[3, :] ** 2), 1).reshape(n, 1)
        return energy

class LJ_L_old(_PriorLayer):
    """Layer for calculating pairwise LJ energy prior.
    """

    def __init__(self, callback_indices, interaction_parameters):
        super(LJ_L_old, self).__init__(
            callback_indices, interaction_parameters)
        for param_dict in self.interaction_parameters:
            if (key in param_dict for key in ('c12', 'c6')):
                pass
            else:
                raise KeyError(
                    'Missing or incorrect key for repulsion parameters'
                )
                
        lj_parameters = torch.tensor([], requires_grad=True)
        for param_dict in self.interaction_parameters:
            lj_parameters = torch.cat((
                lj_parameters,
                torch.sqrt(torch.tensor([[param_dict['c12']],
                              [param_dict['c6']]], requires_grad=True))), dim=1)
        LJ_parameters = nn.Parameter(lj_parameters)
        
        self.register_parameter('LJ_parameters', LJ_parameters)

    def forward(self, in_feat):
        """
        Returns
        -------
        energy: torch.Tensor
            output energy of size (n,1) for n examples.
        """

        n = len(in_feat)
        energy = torch.sum((self.LJ_parameters[0, :] ** 2)/(in_feat ** 12)
                           - (self.LJ_parameters[1, :] ** 2)/ (in_feat ** 6), 1).reshape(n, 1)
        return energy
        
        
class LJ_namd(_PriorLayer):
    """Layer for calculating pairwise LJ energy prior.
    """

    def __init__(self, callback_indices, interaction_parameters, mol = 'fibri'):
        super(LJ_namd, self).__init__(
            callback_indices, interaction_parameters)
        for param_dict in self.interaction_parameters:
            if (key in param_dict for key in ('eps', 'rmin')):
                pass
            else:
                raise KeyError(
                    'Missing or incorrect key for repulsion parameters'
                )
                
        eps = torch.tensor([], requires_grad=True)
        rmin = torch.tensor([], requires_grad=True)
        for param_dict in self.interaction_parameters:
            eps = torch.cat((eps,torch.tensor([[param_dict['eps']]], requires_grad=True)),
                                                           dim=1)
            rmin = torch.cat((rmin,torch.tensor([[np.sqrt(param_dict['rmin'])]], requires_grad=True)),
                                                           dim=1)
        
        eps_parameters = nn.Parameter(eps)
        rmin_parameters = nn.Parameter(rmin)
        
        self.register_parameter('eps_parameters', eps_parameters)
        self.register_parameter('rmin_parameters', rmin_parameters)
        
        self.mol = mol
        self.what = []
        
        if self.mol == 'fibri':
            bond_list = [(0,11), (1,6),(2,3),(2,6),(2,9),(3,4),(4,10),(5,7),(5,14),(8,9),(8,13),(11,12),(12,13),(12,14)]
            for i in range(15):
              for j in range(i+1,15):
                if (i,j) not in bond_list:
                  self.what.append((i,j))
        elif self.mol == 'chi':
            bond_list = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),(6, 7),(7, 8),(8, 9)]
            for i in range(10):
              for j in range(i+1,10):
                if (i,j) not in bond_list:
                  self.what.append((i,j))
        elif self.mol == 'chi_cacb':
            bond_list = [(0,2),(2,4),(4,6),(6,8),(8,10),(10,12),(12,13),(13,15),(15,17),(0,1),(2,3),(4,5),(6,7),(8,9),(10,11),(13,14),(15,16),(17,18)]
            for i in range(19):
              for j in range(i+1,19):
                if (i,j) not in bond_list:
                  self.what.append((i,j))
        else:
            raise RuntimeError('Not supported molecule type!')
        

    def forward(self, in_feat):
        """
        Returns
        -------
        energy: torch.Tensor
            output energy of size (n,1) for n examples.
        """
        
        n = len(in_feat)
        if self.mol == 'fibri':
            # what=[(0, 1),(1, 2),(2, 3),(3, 4),(4, 5),(5, 6),(6, 7),(7, 8),(8, 9),(9, 10),(10, 11),(11, 12),(12, 13),(13, 14),(0, 2),(1, 3),(2, 4),(3, 5),(4, 6),(5, 7),(6, 8),(7, 9),(8, 10),(9, 11),(10, 12),(11, 13),(12, 14),(0, 3),(1, 4),(2, 5),(3, 6),(4, 7),(5, 8),(6, 9),(7, 10),(8, 11),(9, 12),(10, 13),(11, 14),(0, 4),(1, 5),(2, 6),(3, 7),(4, 8),(5, 9),(6, 10),(7, 11),(8, 12),(9, 13),(10, 14),(0, 5),(1, 6),(2, 7),(3, 8),(4, 9),(5, 10),(6, 11),(7, 12),(8, 13),(9, 14),(0, 6),(1, 7),(2, 8),(3, 9),(4, 10),(5, 11),(6, 12),(7, 13),(8, 14),(0, 7),(1, 8),(2, 9),(3, 10),(4, 11),(5, 12),(6, 13),(7, 14),(0, 8),(1, 9),(2, 10),(3, 11),(4, 12),(5, 13),(6, 14),(0, 9),(1, 10),(2, 11),(3, 12),(4, 13),(5, 14),(0, 10),(1, 11),(2, 12),(3, 13),(4, 14),(0, 11),(1, 12),(2, 13),(3, 14),(0, 12),(1, 13),(2, 14),(0, 13),(1, 14),(0, 14)]
            # start = np.zeros((105,15))
            # end = np.zeros((105,15))
            # for i in range(105):
            #   start[i,what[i][0]]=1
            #   end[i,what[i][1]]=1
            start = np.zeros((91,15))
            end = np.zeros((91,15))
            for i in range(91):
              start[i,self.what[i][0]]=1
              end[i,self.what[i][1]]=1
        elif self.mol == 'chi':
            start = np.zeros((36,10))
            end = np.zeros((36,10))
            for i in range(36):
              start[i,self.what[i][0]]=1
              end[i,self.what[i][1]]=1
        elif self.mol == 'chi_cacb':
            start = np.zeros((153,19))
            end = np.zeros((153,19))
            for i in range(153):
              start[i,self.what[i][0]]=1
              end[i,self.what[i][1]]=1
        else:
            raise RuntimeError('Not supported molecule type!')

        start=torch.tensor(start)
        end=torch.tensor(end)
        
        e = torch.reshape(self.eps_parameters,(self.eps_parameters.shape[1],))
        r = torch.reshape(self.rmin_parameters,(self.rmin_parameters.shape[1],))
        energy = torch.sum(torch.sqrt(torch.matmul(start,e)*torch.matmul(end,e))*(((torch.matmul(start,r ** 2)+torch.matmul(end,r ** 2))/2/in_feat)** 12-2*(((torch.matmul(start,r ** 2)+torch.matmul(end,r ** 2))/2/in_feat)** 6)), 1).reshape(n, 1)
        return energy

class LJ_L_cutoff(_PriorLayer):
    """Layer for calculating pairwise LJ energy prior.
    """

    def __init__(self, callback_indices, interaction_parameters):
        super(LJ_L_cutoff, self).__init__(
            callback_indices, interaction_parameters)
        for param_dict in self.interaction_parameters:
            if (key in param_dict for key in ('c12', 'c6', 'a', 'b')):
                pass
            else:
                raise KeyError(
                    'Missing or incorrect key for repulsion parameters'
                )
        self.cutoff = 10.0
        lj_parameters_cutoff = torch.tensor([], requires_grad=True)
        for param_dict in self.interaction_parameters:
            lj_parameters_cutoff = torch.cat((
                lj_parameters_cutoff,
                torch.sqrt(torch.tensor([[param_dict['c12']],
                              [param_dict['c6']],
                              [param_dict['a']],
                              [param_dict['b']]], requires_grad=True))), dim=1)
        lj_parameters_cutoff = nn.Parameter(lj_parameters_cutoff)
        
        self.register_parameter('lj_parameters_cutoff', lj_parameters_cutoff)

    def forward(self, in_feat):
        """
        Returns
        -------
        energy: torch.Tensor
            output energy of size (n,1) for n examples.
        """

        n = len(in_feat)
        
        energy = torch.sum(((self.lj_parameters_cutoff[0, :]/in_feat)
                           ** self.lj_parameters_cutoff[2, :] - (self.lj_parameters_cutoff[1, :]/in_feat)
                           ** (self.lj_parameters_cutoff[2, :]/2)) * (torch.lt(in_feat,self.cutoff).int()), 1).reshape(n, 1)
        return energy

class HarmonicLayer(_PriorLayer):
    """Layer for calculating bond/angle harmonic energy prior. Harominc energy
    contributions have the following form:

        U_harmonic_{ij} = 0.5 * k_{ij} * ((r_{ij} - r_0_{ij}) ^ 2)

    where U_harmonic_ij is the harmonic energy contribution from
    coarse grain beads i and j, k_ij is the harmonic spring constant
    (in energy/distance**2) that characterizes the strength of the harmonic
    interaction between coarse grain beads i and j, r_{ij} is the pairwise
    distance (in distance units) between coarse grain beads i and j, and r_0_ij
    is the equilibrium/average pairwise distance (in distance units) between
    coarse grain beads i and j.

    Parameters
    ----------
    callback_indices: list of int
        indices used to access a specified subset of outputs from the feature
        layer through a residual connection

    interaction_parameters : list of python dictionaries
        list of dictionaries that specify the constants characterizing
        interactions between beads. Each list element corresponds to a single
        interaction using a dictionary of parameters keyed to corresponding
        numerical values. The order of these dictionaries follows the same order
        as the callback indices specifying which outputs from the feature layer
        should pass through the prior. The parameters for HarmonicLayer
        dictionaries are 'mean', the center of the harmonic interaction
        (in length or angle units), and 'k', the (positive) harmonic spring
        constant (in units of energy / length**2 or 1 / length**2).

    Attributes
    ----------
    harmonic_parameters : torch.Tensor
        tensor of shape [2, num_interactions]. The first row contains the
        harmonic spring constants, the second row contains the mean positions,
        and each column corresponds to a single interaction in the order
        determined by self.callback_indices

    Notes
    -----
    This prior energy is useful for constraining the CGnet potential in regions
    of configuration space in which sampling is normally precluded by physical
    harmonic constraints associated with the structural integrity of the protein
    along its backbone. The harmonic parameters are also easily estimated from
    all atom simulation data because bond and angle distributions typically have
    Gaussian structure, which is easily intepretable as a harmonic energy
    contribution via the Boltzmann distribution.

    References
    ----------
    Wang, J., Olsson, S., Wehmeyer, C., PÃ©rez, A., Charron, N. E.,
        de Fabritiis, G., NoÃ©, F., Clementi, C. (2019). Machine Learning
        of Coarse-Grained Molecular Dynamics Force Fields. ACS Central Science.
        https://doi.org/10.1021/acscentsci.8b00913
    """

    def __init__(self, callback_indices, interaction_parameters):
        super(HarmonicLayer, self).__init__(
            callback_indices, interaction_parameters)
        for param_dict in self.interaction_parameters:
            if (key in param_dict for key in ('k', 'mean')):
                if torch.isnan(param_dict['k']).any():
                    raise ValueError(
                    'Harmonic spring constant "k" contains NaNs.' \
                    'Check your parameters.'
                        )
                if torch.isnan(param_dict['mean']).any():
                    raise ValueError(
                    'Center of the harmonic interaction "mean" contains NaNs.'\
                    'Check your parameters.'
                        )
            else:
                KeyError('Missing or incorrect key for harmonic parameters')
        harmonic_parameters = torch.tensor([])
        for param_dict in self.interaction_parameters:
            harmonic_parameters = torch.cat((harmonic_parameters,
                                             torch.tensor([[param_dict['k']],
                                                           [param_dict['mean']]])),
                                                           dim=1)
        self.register_buffer('harmonic_parameters', harmonic_parameters)

    def forward(self, in_feat):
        """Calculates harmonic contribution of bond/angle interactions to energy

        Parameters
        ----------
        in_feat: torch.Tensor
            input features, such as bond distances or angles of size (n,k), for
            n examples and k features.

        Returns
        -------
        energy: torch.Tensor
            output energy of size (n,1) for n examples.

        """

        n = len(in_feat)
        energy = torch.sum(self.harmonic_parameters[0, :] *
                           (in_feat - self.harmonic_parameters[1, :]) ** 2,
                           1).reshape(n, 1) / 2
        return energy

class HarmonicLayer_L(_PriorLayer):
    """
    """

    def __init__(self, callback_indices, interaction_parameters):
        super(HarmonicLayer_L, self).__init__(
            callback_indices, interaction_parameters)
        for param_dict in self.interaction_parameters:
            if (key in param_dict for key in ('k', 'mean')):
                if torch.isnan(param_dict['k']).any():
                    raise ValueError(
                    'Harmonic spring constant "k" contains NaNs.' \
                    'Check your parameters.'
                        )
                if torch.isnan(param_dict['mean']).any():
                    raise ValueError(
                    'Center of the harmonic interaction "mean" contains NaNs.'\
                    'Check your parameters.'
                        )
            else:
                KeyError('Missing or incorrect key for harmonic parameters')
        h_parameters = torch.tensor([], requires_grad=True)
        for param_dict in self.interaction_parameters:
            h_parameters = torch.cat((h_parameters,
                                             torch.sqrt(torch.tensor([[param_dict['k']],
                                                           [param_dict['mean']]], requires_grad=True))),
                                                           dim=1)
        harmonic_parameters = nn.Parameter(h_parameters)
        self.register_parameter('harmonic_parameters', harmonic_parameters)

    def forward(self, in_feat):
        """Calculates harmonic contribution of bond/angle interactions to energy

        Parameters
        ----------
        in_feat: torch.Tensor
            input features, such as bond distances or angles of size (n,k), for
            n examples and k features.

        Returns
        -------
        energy: torch.Tensor
            output energy of size (n,1) for n examples.

        """

        n = len(in_feat)
        energy = torch.sum((self.harmonic_parameters[0, :] ** 2) *
                           (in_feat - (self.harmonic_parameters[1, :] ** 2)) ** 2,
                           1).reshape(n, 1) / 2
        return energy

class HarmonicLayer_L_angle(_PriorLayer):
    """
    """

    def __init__(self, callback_indices, interaction_parameters):
        super(HarmonicLayer_L_angle, self).__init__(
            callback_indices, interaction_parameters)
        for param_dict in self.interaction_parameters:
            if (key in param_dict for key in ('k', 'mean')):
                if torch.isnan(param_dict['k']).any():
                    raise ValueError(
                    'Harmonic spring constant "k" contains NaNs.' \
                    'Check your parameters.'
                        )
                if torch.isnan(param_dict['mean']).any():
                    raise ValueError(
                    'Center of the harmonic interaction "mean" contains NaNs.'\
                    'Check your parameters.'
                        )
            else:
                KeyError('Missing or incorrect key for harmonic parameters')
        h_parameters = torch.tensor([], requires_grad=True)
        for param_dict in self.interaction_parameters:
            h_parameters = torch.cat((h_parameters,
                                             torch.tensor([[np.sqrt(param_dict['k'])],
                                                           [param_dict['mean']]], requires_grad=True)),
                                                           dim=1)
        harmonic_parameters = nn.Parameter(h_parameters)
        self.register_parameter('harmonic_parameters', harmonic_parameters)

    def forward(self, in_feat):
        """Calculates harmonic contribution of bond/angle interactions to energy

        Parameters
        ----------
        in_feat: torch.Tensor
            input features, such as bond distances or angles of size (n,k), for
            n examples and k features.

        Returns
        -------
        energy: torch.Tensor
            output energy of size (n,1) for n examples.

        """

        n = len(in_feat)
        energy = torch.sum((self.harmonic_parameters[0, :] ** 2) *
                           (in_feat - self.harmonic_parameters[1, :]) ** 2,
                           1).reshape(n, 1) / 2
        return energy

class HarmonicLayer_L_k(_PriorLayer):
    """
    """

    def __init__(self, callback_indices, interaction_parameters):
        super(HarmonicLayer_L_k, self).__init__(
            callback_indices, interaction_parameters)
        for param_dict in self.interaction_parameters:
            if (key in param_dict for key in ('k', 'mean')):
                if torch.isnan(param_dict['k']).any():
                    raise ValueError(
                    'Harmonic spring constant "k" contains NaNs.' \
                    'Check your parameters.'
                        )
                if torch.isnan(param_dict['mean']).any():
                    raise ValueError(
                    'Center of the harmonic interaction "mean" contains NaNs.'\
                    'Check your parameters.'
                        )
            else:
                KeyError('Missing or incorrect key for harmonic parameters')
        h_parameters = torch.tensor([], requires_grad=True)
        equilibrium_states = torch.tensor([])
        for param_dict in self.interaction_parameters:
            h_parameters = torch.cat((h_parameters,
                                             torch.tensor([[np.sqrt(param_dict['k'])]], requires_grad=True)),
                                                           dim=1)
            equilibrium_states = torch.cat((equilibrium_states,
                                             torch.tensor([[param_dict['mean']]])),
                                                           dim=1)
                                                           
        self.register_buffer('equilibrium_states', equilibrium_states)
        harmonic_parameters = nn.Parameter(h_parameters)
        self.register_parameter('harmonic_parameters', harmonic_parameters)

    def forward(self, in_feat):
        """Calculates harmonic contribution of bond/angle interactions to energy

        Parameters
        ----------
        in_feat: torch.Tensor
            input features, such as bond distances or angles of size (n,k), for
            n examples and k features.

        Returns
        -------
        energy: torch.Tensor
            output energy of size (n,1) for n examples.

        """

        n = len(in_feat)
        energy = torch.sum((self.harmonic_parameters ** 2) *
                           (in_feat - self.equilibrium_states) ** 2,
                           1).reshape(n, 1) / 2
        return energy


class HarmonicLayer_L_r(_PriorLayer):
    """
    """

    def __init__(self, callback_indices, interaction_parameters):
        super(HarmonicLayer_L_r, self).__init__(
            callback_indices, interaction_parameters)
        for param_dict in self.interaction_parameters:
            if (key in param_dict for key in ('k', 'mean')):
                if torch.isnan(param_dict['k']).any():
                    raise ValueError(
                    'Harmonic spring constant "k" contains NaNs.' \
                    'Check your parameters.'
                        )
                if torch.isnan(param_dict['mean']).any():
                    raise ValueError(
                    'Center of the harmonic interaction "mean" contains NaNs.'\
                    'Check your parameters.'
                        )
            else:
                KeyError('Missing or incorrect key for harmonic parameters')
        h_parameters = torch.tensor([], requires_grad=True)
        spring_factors = torch.tensor([])
        for param_dict in self.interaction_parameters:
            h_parameters = torch.cat((h_parameters,
                                             torch.tensor([[np.sqrt(param_dict['mean'])]], requires_grad=True)),
                                                           dim=1)
            spring_factors = torch.cat((spring_factors,
                                             torch.tensor([[param_dict['k']]])),
                                                           dim=1)
                                                           
        self.register_buffer('spring_factors', spring_factors)
        harmonic_parameters = nn.Parameter(h_parameters)
        self.register_parameter('harmonic_parameters', harmonic_parameters)

    def forward(self, in_feat):
        """Calculates harmonic contribution of bond/angle interactions to energy

        Parameters
        ----------
        in_feat: torch.Tensor
            input features, such as bond distances or angles of size (n,k), for
            n examples and k features.

        Returns
        -------
        energy: torch.Tensor
            output energy of size (n,1) for n examples.

        """

        n = len(in_feat)
        energy = torch.sum(self.spring_factors *
                           (in_feat - (self.harmonic_parameters) ** 2) ** 2,
                           1).reshape(n, 1) / 2
        return energy

class HarmonicLayer_L_r_angle(_PriorLayer):
    """
    """

    def __init__(self, callback_indices, interaction_parameters):
        super(HarmonicLayer_L_r_angle, self).__init__(
            callback_indices, interaction_parameters)
        for param_dict in self.interaction_parameters:
            if (key in param_dict for key in ('k', 'mean')):
                if torch.isnan(param_dict['k']).any():
                    raise ValueError(
                    'Harmonic spring constant "k" contains NaNs.' \
                    'Check your parameters.'
                        )
                if torch.isnan(param_dict['mean']).any():
                    raise ValueError(
                    'Center of the harmonic interaction "mean" contains NaNs.'\
                    'Check your parameters.'
                        )
            else:
                KeyError('Missing or incorrect key for harmonic parameters')
        h_parameters = torch.tensor([], requires_grad=True)
        spring_factors = torch.tensor([])
        for param_dict in self.interaction_parameters:
            h_parameters = torch.cat((h_parameters,
                                             torch.tensor([[param_dict['mean']]], requires_grad=True)),
                                                           dim=1)
            spring_factors = torch.cat((spring_factors,
                                             torch.tensor([[param_dict['k']]])),
                                                           dim=1)
                                                           
        self.register_buffer('spring_factors', spring_factors)
        harmonic_parameters = nn.Parameter(h_parameters)
        self.register_parameter('harmonic_parameters', harmonic_parameters)

    def forward(self, in_feat):
        """Calculates harmonic contribution of bond/angle interactions to energy

        Parameters
        ----------
        in_feat: torch.Tensor
            input features, such as bond distances or angles of size (n,k), for
            n examples and k features.

        Returns
        -------
        energy: torch.Tensor
            output energy of size (n,1) for n examples.

        """

        n = len(in_feat)
        energy = torch.sum(self.spring_factors *
                           (in_feat - self.harmonic_parameters) ** 2,
                           1).reshape(n, 1) / 2
        return energy

class FENE_bond_L(_PriorLayer):
    """Layer for calculating FENE Bond energy
    """

    def __init__(self, callback_indices, interaction_parameters):
        super(FENE_bond_L, self).__init__(
            callback_indices, interaction_parameters)
        for param_dict in self.interaction_parameters:
            if (key in param_dict for key in ('k', 'mean')):
                if torch.isnan(param_dict['k']).any():
                    raise ValueError(
                    'Harmonic spring constant "k" contains NaNs.' \
                    'Check your parameters.'
                        )
                if torch.isnan(param_dict['mean']).any():
                    raise ValueError(
                    'Center of the harmonic interaction "mean" contains NaNs.'\
                    'Check your parameters.'
                        )
            else:
                KeyError('Missing or incorrect key for harmonic parameters')
        fene_parameters = torch.tensor([], requires_grad=True)
        for param_dict in self.interaction_parameters:
            fene_parameters = torch.cat((fene_parameters,
                                             torch.tensor([[param_dict['k']],
                                                           [param_dict['mean']]], requires_grad=True)),
                                                           dim=1)
        FENE_parameters = nn.Parameter(fene_parameters)
        self.register_parameter('FENE_parameters', FENE_parameters)

    def forward(self, in_feat):
        """

        Returns
        -------
        energy: torch.Tensor
            output energy of size (n,1) for n examples.

        """

        n = len(in_feat)
        energy = -torch.sum(self.FENE_parameters[0, :] *
                           (self.FENE_parameters[1, :] ** 2) * torch.log(1-torch.div(in_feat ** 2,self.FENE_parameters[1, :] ** 2)),
                           1).reshape(n, 1) / 2
        return energy


class REB_angle_L(_PriorLayer):
    """Layer for calculating FENE Bond energy
    """

    def __init__(self, callback_indices, interaction_parameters):
        super(REB_angle_L, self).__init__(
            callback_indices, interaction_parameters)
        for param_dict in self.interaction_parameters:
            if (key in param_dict for key in ('k', 'mean')):
                if torch.isnan(param_dict['k']).any():
                    raise ValueError(
                    'Harmonic spring constant "k" contains NaNs.' \
                    'Check your parameters.'
                        )
                if torch.isnan(param_dict['mean']).any():
                    raise ValueError(
                    'Center of the harmonic interaction "mean" contains NaNs.'\
                    'Check your parameters.'
                        )
            else:
                KeyError('Missing or incorrect key for harmonic parameters')
        reb_parameters = torch.tensor([], requires_grad=True)
        equilibrium_states = torch.tensor([])
        for param_dict in self.interaction_parameters:
            reb_parameters = torch.cat((reb_parameters,
                                             torch.tensor([[param_dict['k']]], requires_grad=True)),
                                                           dim=1)
            equilibrium_states = torch.cat((equilibrium_states,
                                             torch.tensor([[param_dict['mean']]])),
                                                           dim=1)
                                                           
        self.register_buffer('equilibrium_states', equilibrium_states)
        REB_parameters = nn.Parameter(reb_parameters)
        self.register_parameter('REB_parameters', REB_parameters)

    def forward(self, in_feat):
        """

        Returns
        -------
        energy: torch.Tensor
            output energy of size (n,1) for n examples.

        """

        n = len(in_feat)
        energy = torch.sum((self.REB_parameters[0, :] **2) *
                           (torch.div(torch.cos(in_feat) - torch.cos(self.equilibrium_states), torch.sin(in_feat))) ** 2,
                           1).reshape(n, 1) / 2
        return energy


class ZscoreLayer(nn.Module):
    """Layer for Zscore normalization. Zscore normalization involves
    scaling features by their mean and standard deviation in the following
    way:

        X_normalized = (X - X_avg) / sigma_X

    where X_normalized is the zscore-normalized feature, X is the original
    feature, X_avg is the average value of the orignal feature, and sigma_X
    is the standard deviation of the original feature.

    Parameters
    ----------
    zscores: torch.Tensor
        [2, n_features] tensor, where the first row contains the means
        and the second row contains the standard deviations of each
        feature

    Notes
    -----
    Zscore normalization can accelerate training convergence if placed
    after a GeometryFeature() layer, especially if the input features
    span different orders of magnitudes, such as the combination of angles
    and distances.

    For more information, see the documentation for
    sklearn.preprocessing.StandardScaler

    """

    def __init__(self, zscores):
        super(ZscoreLayer, self).__init__()
        self.register_buffer('zscores', zscores)

    def forward(self, in_feat):
        """Normalizes each feature by subtracting its mean and dividing by
           its standard deviation.

        Parameters
        ----------
        in_feat: torch.Tensor
            input data of shape [n_frames, n_features]

        Returns
        -------
        rescaled_feat: torch.Tensor
            Zscore normalized features. Shape [n_frames, n_features]

        """
        rescaled_feat = (in_feat - self.zscores[0, :])/self.zscores[1, :]
        return rescaled_feat

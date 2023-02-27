# Authors: Nick Charron, Dominik Lemm
# Contributors: Brooke Husic


import numpy as np
import torch
import torch.nn as nn


class ShiftedSoftplus(nn.Module):
    r""" Shifted softplus (SSP) activation function

    SSP originates from the softplus function:

        y = \ln\left(1 + e^{-x}\right)

    Schütt et al. (2018) introduced a shifting factor to the function in order
    to ensure that SSP(0) = 0 while having infinite order of continuity:

         y = \ln\left(1 + e^{-x}\right) - \ln(2)

    SSP allows to obtain smooth potential energy surfaces and second derivatives
    that are required for training with forces as well as the calculation of
    vibrational modes (Schütt et al. 2018).

    References
    ----------
    K.T. Schütt. P.-J. Kindermans, H. E. Sauceda, S. Chmiela,
        A. Tkatchenko, K.-R. Müller. (2018)
        SchNet - a deep learning architecture for molecules and materials.
        The Journal of Chemical Physics.
        https://doi.org/10.1063/1.5019779

    """

    def __init__(self):
        super(ShiftedSoftplus, self).__init__()

    def forward(self, input_tensor):
        """ Applies the shifted softplus function element-wise

        Parameters
        ----------
        input_tensor: torch.Tensor
            Input tensor of size (n_examples, *) where `*` means, any number of
            additional dimensions.

        Returns
        -------
        Output: torch.Tensor
            Same size (n_examples, *) as the input.
        """
        return nn.functional.softplus(input_tensor) - np.log(2.0)


class _AbstractRBFLayer(nn.Module):
    """Abstract layer for definition of radial basis function layers"""

    def __init__(self):
        super(_AbstractRBFLayer, self).__init__()

    def __len__(self):
        """Method to get the size of the basis used for distance expansions.

        Notes
        -----
        This method must be implemented explicitly in a child class. If not,
        a NotImplementedError will be raised
        """
        raise NotImplementedError()

    def forward(self, distances):
        """Forward method to compute expansions of distances into basis
        functions.

        Notes
        -----
        This method must be explicitly implemented in a child clase.
        If not, a NotImplementedError will be raised.
        """
        raise NotImplementedError()


class GaussianRBF(_AbstractRBFLayer):
    r"""Radial basis function (RBF) layer

    This layer serves as a distance expansion using radial basis functions with
    the following form:

        e_k (r_j - r_i) = exp(- (\left \| r_j - r_i \right \| - \mu_k)^2 / (2 * var)

    with centers mu_k calculated on a uniform grid between
    zero and the distance cutoff and var as the variance.
    The radial basis function has the effect of decorrelating the
    convolutional filter, which improves the training time. All distances are
    assumed, by default, to have units of Angstroms.

    Parameters
    ----------
    low_cuttof : float (default=0.0)
        Minimum distance cutoff for the Gaussian basis. This cuttoff represents the
        center of the first basis funciton.
    high_cutoff : float (default=5.0)
        Maximum distance cutoff for the Gaussian basis. This cuttoff represents the
        center of the last basis function.
    n_gaussians : int (default=50)
        Total number of Gaussian functions to calculate. Number will be used to
        create a uniform grid from 0.0 to cutoff. The number of Gaussians will
        also decide the output size of the RBF layer output
        ([n_examples, n_beads, n_neighbors, n_gauss]). The default number of
        gaussians is the same as that in SchnetPack (Schutt et al, 2019).
    variance : float (default=1.0)
        The variance (standard deviation squared) of the Gaussian functions.
    normalize_output : bool (default=False)
        If True, the output of the GaussianRBF layer will be normalized by the sum
        over the outputs from every basis function.

    Example
    -------
    To instance a SchnetFeature using a GaussianRBF layer with 50 centers, a
    low cuttof of 1 distance unit, a high cutoff of 50 distance units, a
    variance of 0.8, and no output normalization, the following procedure can
    be used:

        rbf_layer = GaussianRBF(low_cutoff=1.0, high_cutoff=50.0,
                                n_gaussians=50, variance=0.8)
        schnet_feature = SchnetFeature(feature_size = ...,
                                       embedding_layer = ...,
                                       rbf_layer=rbf_layer,
                                       n_interaction_blocks = ...,
                                       calculate_geometry = ...,
                                       n_beads = ...,
                                       neighbor_cutoff = ...,
                                       device = ...)

    where the elipses represent the other parameters of the SchnetFeature that
    are specific to your needs (see cgnet.feature.SchnetFeature for more
    details).

    Notes
    -----
    The units of the variance and cutoffs are fixed by the units of the
    input distances.

    References
    ----------
    Schutt, K. T., Kessel, P., Gastegger, M., Nicoli, K. A., Tkatchenko, A.,
         & Müller, K.-R. (2019). SchNetPack: A Deep Learning Toolbox For Atomistic
         Systems. Journal of Chemical Theory and Computation, 15(1), 448–455.
         https://doi.org/10.1021/acs.jctc.8b00908
    """

    def __init__(self, low_cutoff=0.0, high_cutoff=5.0, n_gaussians=50,
                 variance=1.0, normalize_output=False):
        super(GaussianRBF, self).__init__()
        self.register_buffer('centers', torch.linspace(low_cutoff,
                             high_cutoff, n_gaussians))
        self.variance = variance
        self.normalize_output = normalize_output

    def __len__(self):
        """Method to return basis size"""
        return len(self.centers)

    def forward(self, distances, distance_mask=None):
        """Calculate Gaussian expansion

        Parameters
        ----------
        distances : torch.Tensor
            Interatomic distances of size [n_examples, n_beads, n_neighbors]
        distance_mask : torch.Tensor
            Mask of shape [n_examples, n_beads, n_neighbors] to filter out
            contributions from non-physical beads introduced from padding
            examples from molecules with varying sizes

        Returns
        -------
        gaussian_exp: torch.Tensor
            Gaussian expansions of size [n_examples, n_beads, n_neighbors,
            n_gauss]
        """
        dist_centered_squared = torch.pow(distances.unsqueeze(dim=3) -
                                          self.centers, 2)
        gaussian_exp = torch.exp(-(0.5 / self.variance)
                                 * dist_centered_squared)

        # If specified, normalize output by sum over all basis function outputs
        if self.normalize_output:
            basis_sum = torch.sum(gaussian_exp, dim=3)
            gaussian_exp = gaussian_exp / basis_sum[:, :, :, None]

        # Mask the output of the radial distribution with the distance mask
        if distance_mask is not None:
            gaussian_exp = gaussian_exp * distance_mask[:, :, :, None]
        return gaussian_exp


class PolynomialCutoffRBF(_AbstractRBFLayer):
    r"""Radial basis function (RBF) layer
    This layer serves as a distance expansion using modulated radial
    basis functions with the following form:

        g_k(r_{ij}) = \phi(r_{ij}, cutoff) *
        exp(- \beta * (\left \exp(\alpha * -r_{ij}) - \mu_k\right)^2)

    where \phi(r_{ij}, cutoff) is a piecewise polynomial modulation
    function of the following form,

                /
               |    1 - 6*(r_{ij}/cutoff)^5
               |    + 15*(r_{ij}/cutoff)^4      for r_{ij} < cutoff
     \phi = -- |    - 10*(r_{ij}/cutoff)^3
               |
               |    0.0                         for r_{ij} >= cutoff
                \

    the centers mu_k calculated on a uniform grid between
    exp(-low_cutoff) and exp(-high_cutoff), and beta as a scaling
    parameter defined as:

        \beta = ((2/n_gaussians) * (1 - exp(-cutoff))^-2

    The radial basis function has the effect of decorrelating the
    convolutional filter, which improves the training time. All distances
    are assumed, by default, to have units of Angstroms. we suggest that
    users visually inspect their basis before use in order to make sure
    that they are satisfied with the distribution and cutoffs of the
    functions.

    Parameters
    ----------
    low_cutoff : float (default=0.0)
        Low distance cutoff for the modulation. This parameter,
        along with high_cutoff, determine the distribution of the centers of
        each basis function.
    high_cutoff : float (default=10.0)
        Distance cutoff for the modulation. This parameter,
        along with low_cutoff, determine the distribution of centers of
        each basis function.
    alpha : float (default=1.0)
        This parameter is a prefactor to the following term:

                         alpha * exp(-r_ij)

        Lower values of this parameter results in a slower transition between
        sharply peaked gaussian functions at smaller distances and broadly peaked
        gaussian functions at larger distances.
        with slowly decaying tails.
    n_gaussians : int (default=64)
        Total number of gaussian functions to calculate. Number will be used to
        create a uniform grid from exp(-cutoff) to 1. The number of gaussians
        will also decide the output size of the RBF layer output
        ([n_examples, n_beads, n_neighbors, n_gauss]). The default value of
        64 gaussians is taken from Unke & Meuwly (2019).
    normalize_output : bool (default=False)
        If True, the output of the PolynomialCutoffRBF layer will be normalized
        by the sum over the outputs from every basis function.
    tolerance : float (default=1e-10)
        When expanding the modulated gaussians, values below the tolerance
        will be set to zero.
    device : torch.device (default=torch.device('cpu'))
        Device upon which tensors are mounted

    Attributes
    ----------
    beta : float
        Gaussian decay parameter, defined as:
            \beta = ((2/n_gaussians) * (1 - exp(-cutoff))^-2

    Example
    -------
    To instance a SchnetFeature using a PolynomialCutoffRBF layer with 50 centers,
    a low cuttof of 1 distance unit, a high cutoff of 50 distance units, an
    alpha value of 0.8, and no output normalization, the following procedure can
    be used:

        rbf_layer = PolynomialCutoffRBF(low_cutoff=1.0, high_cutoff=50.0,
                                        n_gaussians=50, variance=0.8)
        schnet_feature = SchnetFeature(feature_size = ...,
                                       embedding_layer = ...,
                                       rbf_layer=rbf_layer,
                                       n_interaction_blocks = ...,
                                       calculate_geometry = ...,
                                       n_beads = ...,
                                       neighbor_cutoff = ...,
                                       device = ...)

    where the elipses represent the other parameters of the SchnetFeature that
    are specific to your needs (see cgnet.feature.SchnetFeature for more
    details).


    Notes
    -----
    These basis functions were originally introduced as part of the PhysNet
    architecture (Unke & Meuwly, 2019). Though the basis function centers are
    scattered uniformly, the modulation function has the effect of broadening
    those functions closer to the specified cutoff. The overall result is a set
    of basis functions which have high resolution at small distances which
    smoothly morphs to basis functions with lower resolution at larger
    distances.

    The units of the variance, cutoffs, alpha, and beta are fixed by the units
    of the input distances.

    References
    ----------
    Unke, O. T., & Meuwly, M. (2019). PhysNet: A Neural Network for Predicting
        Energies, Forces, Dipole Moments and Partial Charges. Journal of
        Chemical Theory and Computation, 15(6), 3678–3693.
        https://doi.org/10.1021/acs.jctc.9b00181

    """

    def __init__(self, low_cutoff=0.0, high_cutoff=10.0, alpha=1.0,
                 n_gaussians=64, normalize_output=False, tolerance=1e-10,
                 device=torch.device('cpu')):
        super(PolynomialCutoffRBF, self).__init__()
        self.tolerance = tolerance
        self.device = device
        self.register_buffer('centers', torch.linspace(np.exp(-high_cutoff),
                             np.exp(-low_cutoff), n_gaussians))
        self.high_cutoff = high_cutoff
        self.low_cutoff = low_cutoff
        self.beta = np.power(((2/n_gaussians) *
                             (1-np.exp(-self.high_cutoff))), -2)
        self.alpha = alpha
        self.normalize_output = normalize_output

    def __len__(self):
        """Method to return basis size"""
        return len(self.centers)

    def modulation(self, distances):
        """PhysNet cutoff modulation function

        Parameters
        ----------
        distances : torch.Tensor
            Interatomic distances of size [n_examples, n_beads, n_neighbors]

        Returns
        -------
        mod : torch.Tensor
            The modulation envelope of the radial basis functions. Shape
            [n_examples, n_beads, n_neighbors]

        """
        zeros = torch.zeros_like(distances).to(self.device)
        modulation_envelope = torch.where(distances < self.high_cutoff,
                                          1 - 6 *
                                          torch.pow((distances/self.high_cutoff),
                                                    5)
                                          + 15 *
                                          torch.pow((distances/self.high_cutoff),
                                                     4)
                                          - 10 *
                                          torch.pow(
                                              (distances/self.high_cutoff), 3),
                                          zeros)
        return modulation_envelope

    def forward(self, distances, distance_mask=None):
        """Calculate modulated gaussian expansion

        Parameters
        ----------
        distances : torch.Tensor
            Interatomic distances of size [n_examples, n_beads, n_neighbors]
        distance_mask : torch.Tensor
            Mask of shape [n_examples, n_beads, n_neighbors] to filter out
            contributions from non-physical beads introduced from padding
            examples from molecules with varying sizes

        Returns
        -------
        expansions : torch.Tensor
            Modulated gaussian expansions of size
            [n_examples, n_beads, n_neighbors, n_gauss]

        Notes
        -----
        The gaussian portion of the basis function is a function of
        exp(-r_{ij}), not r_{ij}

        """
        dist_centered_squared = torch.pow(torch.exp(self.alpha *
                                          - distances.unsqueeze(dim=3))
                                          - self.centers, 2)
        gaussian_exp = torch.exp(-self.beta
                                 * dist_centered_squared)
        modulation_envelope = self.modulation(distances).unsqueeze(dim=3)

        expansions = modulation_envelope * gaussian_exp

        # In practice, this gives really tiny numbers. For numbers below the
        # tolerance, we just set them to zero.
        expansions = torch.where(torch.abs(expansions) > self.tolerance,
                                 expansions,
                                 torch.zeros_like(expansions))

        # If specified, normalize output by sum over all basis function outputs
        if self.normalize_output:
            basis_sum = torch.sum(expansions, dim=3)
            expansions = expansions / basis_sum[:, :, :, None]

        if distance_mask is not None:
            expansions = expansions * distance_mask[:, :, :, None]
        return expansions


def LinearLayer(
        d_in,
        d_out,
        bias=True,
        activation=None,
        dropout=0,
        weight_init='xavier',
        weight_init_args=None,
        weight_init_kwargs=None):
    r"""Linear layer function

    Parameters
    ----------
    d_in : int
        input dimension
    d_out : int
        output dimension
    bias : bool (default=True)
        specifies whether or not to add a bias node
    activation : torch.nn.Module() (default=None)
        activation function for the layer
    dropout : float (default=0)
        if > 0, a dropout layer with the specified dropout frequency is
        added after the activation.
    weight_init : str, float, or nn.init function (default=\'xavier\')
        specifies the initialization of the layer weights. For non-option
        initializations (eg, xavier initialization), a string may be used
        for simplicity. If a float or int is passed, a constant initialization
        is used. For more complicated initializations, a torch.nn.init function
        object can be passed in.
    weight_init_args : list or tuple (default=None)
        arguments (excluding the layer.weight argument) for a torch.nn.init
        function.
    weight_init_kwargs : dict (default=None)
        keyword arguements for a torch.nn.init function

    Returns
    -------
    seq : list of torch.nn.Module() instances
        the full linear layer, including activation and optional dropout.

    Example
    -------
    MyLayer = LinearLayer(5, 10, bias=True, activation=nn.Softplus(beta=2),
                          weight_init=nn.init.kaiming_uniform_,
                          weight_init_kwargs={"a":0, "mode":"fan_out",
                          "nonlinearity":"leaky_relu"})

    Produces a linear layer with input dimension 5, output dimension 10, bias
    inclusive, followed by a beta=2 softplus activation, with the layer weights
    intialized according to kaiming uniform procedure with preservation of weight
    variance magnitudes during backpropagation.

    """

    seq = [nn.Linear(d_in, d_out, bias=bias)]
    if activation:
        if isinstance(activation, nn.Module):
            seq += [activation]
        else:
            raise TypeError(
                'Activation {} is not a valid torch.nn.Module'.format(
                    str(activation))
            )
    if dropout:
        seq += [nn.Dropout(dropout)]

    with torch.no_grad():
        if weight_init == 'xavier':
            torch.nn.init.xavier_uniform_(seq[0].weight)
        if weight_init == 'identity':
            torch.nn.init.eye_(seq[0].weight)
        if weight_init not in ['xavier', 'identity', None]:
            if isinstance(weight_init, int) or isinstance(weight_init, float):
                torch.nn.init.constant_(seq[0].weight, weight_init)
            if callable(weight_init):
                if weight_init_args is None:
                    weight_init_args = []
                if weight_init_kwargs is None:
                    weight_init_kwargs = []
                weight_init(seq[0].weight, *weight_init_args,
                            **weight_init_kwargs)
            else:
                raise RuntimeError(
                    'Unknown weight initialization \"{}\"'.format(
                        str(weight_init))
                )
    return seq

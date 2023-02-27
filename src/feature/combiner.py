# Authors: Nick Charron

import torch.nn as nn
from cgnet.feature import (GeometryFeature, Geometry, SchnetFeature,
                           GaussianRBF)
import warnings
g = Geometry(method='torch')


class FeatureCombiner(nn.Module):
    """Class for combining GeometryFeatures and SchnetFeatures


    Parameters
    ----------
    layer_list : list of nn.Module objects
        feature layers with which data is transformed before being passed to
        densely/fully connected layers prior to sum pooling and energy
        prediction/force generation.
    save_geometry : boolean (default=True)
        specifies whether or not to save the output of GeometryFeature
        layers. It is important to set this to true if CGnet priors
        are to be used, and need to callback to GeometryFeature outputs.
    propagate_geometry : boolean (default=False)
        specifies whether or not to concatenate geometry features (i.e.,
        distances, angles, and/or dihedrals) to the feature that is
        propagated through the neural network. This is designed to be
        used ONLY when the layer list is a [GeometryFeature, SchnetFeature].
        (default=False)
    distance_indices : list or np.ndarray of int (default=None)
        Indices of distances output from a GeometryFeature layer, used
        to isolate distances for redundant re-indexing for Schnet utilities


    Attributes
    ----------
    layer_list : nn.ModuleList
        feature layers with which data is transformed before being passed to
        densely/fully connected layers prior to sum pooling and energy
        prediction/force generation. The length of layer_list is the number
        of layers.
    interfeature_transforms : list of None or method types
        inter-feature transforms that may be needed during the forward
        method. These functions take the output of a previous feature layer and
        transform/reindex it so that it may be used as input for the next
        feature layer. The length of this list is equal to the length of the
        layer_list. For example, SchnetFeature tools require a redundant form
        for distances, so outputs from a previous GeometryFeature layer must be
        re-indexed.
    save_geometry  : boolean (default=True)
        specifies whether or not to save the output of GeometryFeature
        layers. It is important to set this to true if CGnet priors
        are to be used, and need to callback to GeometryFeature outputs.
    transform_dictionary : dictionary of strings
        dictionary of mappings to provide for specified inter-feature
        transforms. Keys are strings which describe the mapping, and values
        are mapping objects. For example, a redundant distance mapping may be
        represented as:

            {'redundant_distance_maping' : self.redundant_distance_mapping}

    Notes
    -----
    There are several cases for combinations of GeometryFeature and
    SchnetFeature.

    (1) Geometry Feature alone
    (2) Geometry Feature followed by SchnetFeature
    (3) SchnetFeature alone

    (1) corresponds to classic CGnet architecture, as proposed by
    Wang et. al. (2019).

    (2) is a general combination that allows for prior callbacks to
    non-distance features to add functional energy constraints (e.g.,
    angle/bond/repulsion constraints) that supplement the classic SchNet
    architecture as proposed by Schutt et. al. (2018). In this case,
    SchnetFeature should be initialized with calculate_geometry=False,
    as the preceding GeometryFeature layer already computes a geometrical
    featurization. In this case, propagate_geometry can be set to True
    or False depending on whether the geometry features should or should
    not be propagated through the neural network, respectively.

    (3) corresponds to classic pairwise distance-based SchNet. In this case,
    the SchnetFeature must be initialized with calculate_geometry=True
    so that it can use Geometry() tools to calculate distances on the fly.
    If calculate_geometry=False, the input to the network must be pairwise
    distances of size [n_frames, n_beads, n_neighbors], and the terminal
    CGnet autograd function will compute derivates with respect to pairwise
    distances instead of cartesian coordinates.


    References
    ----------
    Wang, J., Olsson, S., Wehmeyer, C., Pérez, A., Charron, N. E.,
        de Fabritiis, G., Noé, F., Clementi, C. (2019). Machine Learning
        of Coarse-Grained Molecular Dynamics Force Fields. ACS Central Science.
        https://doi.org/10.1021/acscentsci.8b00913
    K.T. Schütt. P.-J. Kindermans, H. E. Sauceda, S. Chmiela,
        A. Tkatchenko, K.-R. Müller. (2018)
        SchNet - a deep learning architecture for molecules and materials.
        The Journal of Chemical Physics.
        https://doi.org/10.1063/1.5019779
    """

    def __init__(self, layer_list, save_geometry=True, propagate_geometry=False,
                 distance_indices=None):
        super(FeatureCombiner, self).__init__()
        self.layer_list = nn.ModuleList(layer_list)
        self.save_geometry = save_geometry
        self.propagate_geometry = propagate_geometry
        self.interfeature_transforms = []
        self.transform_dictionary = {}
        self.distance_indices = distance_indices
        _has_schnet = False
        for layer in self.layer_list:
            if isinstance(layer, SchnetFeature):
                _has_schnet = True
                if (layer.calculate_geometry and any(isinstance(layer,
                    GeometryFeature) for layer in self.layer_list)):
                    warnings.warn("This SchnetFeature has been set to "
                                   "calculate pairwise distances. Set "
                                   "SchnetFeature.calculate_geometry=False if you are "
                                   "preceding this SchnetFeature with a GeometryFeature "
                                   "in order to prevent unnecessarily repeated pairwsie "
                                   "distance calculations")
                    self.interfeature_transforms.append(None)
                if (not layer.calculate_geometry and not any(isinstance(layer,
                    GeometryFeature) for layer in self.layer_list)):
                    warnings.warn("This SchnetFeature has not been designated "
                                  "to calculate pairwise distances, but no "
                                  "GeometryFeature was found in the layer "
                                  "list. Please ensure that network input is "
                                  "formulated as pairwise distances.")
                elif layer.calculate_geometry:
                    self.interfeature_transforms.append(None)
                else:
                    if self.distance_indices is None:
                        raise RuntimeError(("Distance indices must be "
                                            "supplied to FeatureCombiner "
                                            "for redundant re-indexing."))
                    self.transform_dictionary['redundant_distance_mapping'] = (
                        g.get_redundant_distance_mapping(layer._distance_pairs))
                    self.interfeature_transforms.append([self.distance_reindex])
                if isinstance(layer, GeometryFeature):
                    if _has_schnet:
                        raise RuntimeError(
                            "A GeometryFeature should never come after a SchnetFeature"
                            )
            else:
                self.interfeature_transforms.append(None)

            # The following just checks whether the layer_list is
            # [GeometryFeature, SchnetFeature] is propagate_geometry
            # is set to true.
            if self.propagate_geometry:
                if len(self.layer_list) != 2:
                      raise RuntimeError(
                        "propagate_geometry is only designed for a layer " \
                        "list of [GeometryFeature, SchnetFeature]"
                        )
                elif not (isinstance(self.layer_list[0], GeometryFeature)
                          and isinstance(self.layer_list[1], SchnetFeature)):
                          raise RuntimeError(
                            "propagate_geometry is only designed for a layer " \
                            "list of [GeometryFeature, SchnetFeature]"
                            )
    
    def distance_reindex(self, geometry_output):
        """Reindexes GeometryFeature distance outputs to redundant form for
        SchnetFeatures and related tools. See
        Geometry.get_redundant_distance_mapping

        Parameters
        ----------
        geometry_output : torch.Tensor
            geometrical feature output frome a GeometryFeature layer, of size
            [n_frames, n_features].

        Returns
        -------
        redundant_distances : torch.Tensor
            pairwise distances transformed to shape
            [n_frames, n_beads, n_beads-1].
        """
        distances = geometry_output[:, self.distance_indices]
        return distances[:, self.transform_dictionary['redundant_distance_mapping']]

    def forward(self, coords, embedding_property=None):
        """Forward method through specified feature layers. The forward
        operation proceeds through self.layer_list in that same order
        that was passed to the FeatureCombiner.

        Parameters
        ----------
        coords : torch.Tensor
            Input cartesian coordinates of size [n_frames, n_beads, 3]
        embedding_property : torch.Tensor (default=None)
            Some property that should be embedded. Can be nuclear charge
            or maybe an arbitrary number assigned for amino-acids.
            Size [n_frames, n_properties].

        Returns
        -------
        feature_ouput : torch.Tensor
            output tensor, of shape [n_frames, n_features] after featurization
            through the layers contained in self.layer_list.
        geometry_features : torch.Tensor (default=None)
            if save_geometry is True and the layer list is not just a single
            GeometryFeature layer, the output of the last GeometryFeature
            layer is returned alongside the terminal features for prior energy
            callback access. Else, None is returned.
        """
        feature_output = coords
        geometry_features = None
        for num, (layer, transform) in enumerate(zip(self.layer_list,
                                                 self.interfeature_transforms)):
            if transform != None:
                # apply transform(s) before the layer if specified
                for sub_transform in transform:
                    feature_output = sub_transform(feature_output)
            if isinstance(layer, SchnetFeature):
                feature_output = layer(feature_output, embedding_property)
            else:
                feature_output = layer(feature_output)
            if isinstance(layer, GeometryFeature) and self.save_geometry:
                geometry_features = feature_output
        return feature_output, geometry_features

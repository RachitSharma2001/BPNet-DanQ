import numpy as np
import keras.layers as kl
from keras.optimizers import Adam
from keras.models import Model
from concise.utils.helper import get_from_module
import bpnet
import bpnet.losses as blosses
import gin
import keras

'''
seqmodel.py - defines the generic BPNet framework that builds off Keras models
    Defines the body, head structure but does not define any of the specific architecture(meaning specific neural networks)
BPNet.py - has an attribute self.seqmodel which it then uses in all of its predict methods: in each of the predict
    methods it runs output = self.seqmodel(input)

this class: Defines the specific architecture of BPNet.py's self.seqmodel attribute
    Defines the 10 convolutional layers of the body, the deconvolutional + fully connected layers of the heads.
'''


# TODO - setup the following model as a simple bpnet (?)

@gin.configurable
def bpnet_model(tasks,
                filters,
                n_dil_layers,
                conv1_kernel_size,
                tconv_kernel_size,
                b_loss_weight=1,
                c_loss_weight=1,
                p_loss_weight=1,
                c_splines=0,
                b_splines=20,
                merge_profile_reg=False,
                lr=0.004,
                tracks_per_task=2,
                padding='same',
                batchnorm=False,
                use_bias=False,
                n_bias_tracks=2,
                profile_metric=None,
                count_metric=None,
                profile_bias_window_sizes=[1, 50],
                seqlen=None,
                skip_type='residual'):
    """Setup the BPNet model architecture

    Args:
      tasks: list of tasks
      filters: number of convolutional filters to use at each layer
      n_dil_layers: number of dilated convolutional filters to use
      conv1_kernel_size: kernel_size of the first convolutional layer
      tconv_kernel_size: kernel_size of the transpose/de-convolutional final layer
      b_loss_weight: binary classification weight
      c_loss_weight: total count regression weight
      p_loss_weight: profile regression weight
      c_splines: number of splines to use in the binary classification output head
      p_splines: number of splines to use in the profile regression output head (0=None)
      merge_profile_reg: if True, total count and profile prediction will be part of
        a single profile output head
      lr: learning rate of the Adam optimizer
      padding: padding in the convolutional layers
      batchnorm: if True, add Batchnorm after every layer. Note: this may mess up the
        DeepLIFT contribution scores downstream
      use_bias: if True, correct for the bias
      n_bias_tracks: how many bias tracks to expect (for both total count and profile regression)
      seqlen: sequence length.
      skip_type: skip connection type ('residual' or 'dense')

    Returns:
      bpnet.seqmodel.SeqModel
    """
    from bpnet.seqmodel import SeqModel
    # In here they import the DilatedConv1D class from /BPNet/layers.py
    # So essentially they define their own kind of dilated convolution layer
    # which is really made up of 1 regular keras convolutional layer and several
    # dilated ones(specified in this class)
    from bpnet.layers import DilatedConv1D, DeConv1D, GlobalAvgPoolFCN, MovingAverages
    from bpnet.metrics import BPNetMetricSingleProfile, default_peak_pred_metric
    from bpnet.heads import ScalarHead, ProfileHead
    from bpnet.metrics import ClassificationMetrics, RegressionMetrics
    from bpnet.losses import multinomial_nll, CountsMultinomialNLL
    import bpnet.losses as bloss
    from bpnet.activations import clipped_exp
    from bpnet.functions import softmax

    assert p_loss_weight >= 0
    assert c_loss_weight >= 0
    assert b_loss_weight >= 0

    # import ipdb
    # ipdb.set_trace()

    # TODO is it possible to re-instantiate the class to get rid of gin train?

    if profile_metric is None:
        print("Using the default profile prediction metric")
        profile_metric = default_peak_pred_metric

    if count_metric is None:
        print("Using the default regression prediction metrics")
        count_metric = RegressionMetrics()

    # Heads -------------------------------------------------
    heads = []
    # Profile prediction
    if p_loss_weight > 0:
        if not merge_profile_reg:
            heads.append(ProfileHead(target_name='{task}/profile',
                                     net=DeConv1D(n_tasks=tracks_per_task,
                                                  filters=filters,
                                                  tconv_kernel_size=tconv_kernel_size,
                                                  padding=padding,
                                                  n_hidden=0,
                                                  batchnorm=batchnorm
                                                  ),
                                     loss=multinomial_nll,
                                     loss_weight=p_loss_weight,
                                     postproc_fn=softmax,
                                     use_bias=use_bias,
                                     bias_input='bias/{task}/profile',
                                     bias_shape=(None, n_bias_tracks),
                                     bias_net=MovingAverages(window_sizes=profile_bias_window_sizes),
                                     metric=profile_metric
                                     ))
        else:
            heads.append(ProfileHead(target_name='{task}/profile',
                                     net=DeConv1D(n_tasks=tracks_per_task,
                                                  filters=filters,
                                                  tconv_kernel_size=tconv_kernel_size,
                                                  padding=padding,
                                                  n_hidden=1,  # use 1 hidden layer in that case
                                                  batchnorm=batchnorm
                                                  ),
                                     activation=clipped_exp,
                                     loss=CountsMultinomialNLL(c_task_weight=c_loss_weight),
                                     loss_weight=p_loss_weight,
                                     bias_input='bias/{task}/profile',
                                     use_bias=use_bias,
                                     bias_shape=(None, n_bias_tracks),
                                     bias_net=MovingAverages(window_sizes=profile_bias_window_sizes),
                                     metric=BPNetMetricSingleProfile(count_metric=count_metric,
                                                                     profile_metric=profile_metric)
                                     ))
            c_loss_weight = 0  # don't need to use the other count loss

    # Count regression
    if c_loss_weight > 0:
        heads.append(ScalarHead(target_name='{task}/counts',
                                net=GlobalAvgPoolFCN(n_tasks=tracks_per_task,
                                                     n_splines=c_splines,
                                                     batchnorm=batchnorm),
                                activation=None,
                                loss='mse',
                                loss_weight=c_loss_weight,
                                bias_input='bias/{task}/counts',
                                use_bias=use_bias,
                                bias_shape=(n_bias_tracks, ),
                                metric=count_metric,
                                ))

    # Binary classification
    if b_loss_weight > 0:
        heads.append(ScalarHead(target_name='{task}/class',
                                net=GlobalAvgPoolFCN(n_tasks=1,
                                                     n_splines=b_splines,
                                                     batchnorm=batchnorm),
                                activation='sigmoid',
                                loss='binary_crossentropy',
                                loss_weight=b_loss_weight,
                                metric=ClassificationMetrics(),
                                ))
    # -------------------------------------------------
    # They create the entire model, with the body and 4 output heads.
    # The body has n_dil_layers + 1 number of layers,
    # in which the first layer is a convolutional layer and the next
    # n_dil_layers layers are dilated.
    # They use adam optimizer and input the transcription factors 
    m = SeqModel(
        body=DilatedConv1D(filters=filters,
                           conv1_kernel_size=conv1_kernel_size,
                           n_dil_layers=n_dil_layers,
                           padding=padding,
                           batchnorm=batchnorm,
                           skip_type=skip_type),
        heads=heads,
        tasks=tasks,
        optimizer=Adam(lr=lr),
        seqlen=seqlen,
    )
    return m


@gin.configurable
def binary_seq_model(tasks,
                     net_body,
                     net_head,
                     lr=0.004,
                     seqlen=None):
    """NOTE: This doesn't work with gin-train since
    the classes injected by gin-config can't be pickled.

    Instead, I created `basset_seq_model`

    ```
    Can't pickle <class 'bpnet.layers.BassetConv'>: it's not the same
    object as bpnet.layers.BassetConv
    ```

    """
    from bpnet.seqmodel import SeqModel
    from bpnet.heads import ScalarHead, ProfileHead
    from bpnet.metrics import ClassificationMetrics
    # Heads -------------------------------------------------
    heads = [ScalarHead(target_name='{task}/class',
                        net=net_head,
                        activation='sigmoid',
                        loss='binary_crossentropy',
                        metric=ClassificationMetrics(),
                        )]
    # -------------------------------------------------
    m = SeqModel(
        body=net_body,
        heads=heads,
        tasks=tasks,
        optimizer=Adam(lr=lr),
        seqlen=seqlen,
    )
    return m


def get(name):
    return get_from_module(name, globals())

# ==================================================================
# model.gin
# --------------------------------------------
# model
import bpnet
import bpnet.models
import bpnet.heads
import bpnet.layers
import bpnet.seqmodel
import bpnet.trainers
import bpnet.losses
import bpnet.datasets
import bpnet.metrics
import bpnet.configurables

# SeqModel
# My guess: The guys who made bpnet created their own class SeqModel which builds on top of keras's models
# So if we were to modify the architecture of bpnet for r-loops, we would modify the specific seqmodel not the keras model
# Here - we create a new SeqModel, and specify the hyperparameters
# What are %seq_width and %tasks?
train.model = @SeqModel()
SeqModel.seqlen = %seq_width
SeqModel.tasks = %tasks
SeqModel.optimizer = @keras.optimizers.Adam()
# The body takes as input one hot vector representing the nucleotides
# and runs it through an algorithm that creates an embedding of the input
# The head then takes as input that embedding and runs it through the rest of the model
# Whats the point of seperating these two parts? For organization?
SeqModel.heads = [@ProfileHead(), @ScalarHead()]  # Heads
SeqModel.body = @DilatedConv1D()  # Body
keras.optimizers.Adam.lr = %lr

# - Body
# Specifying the embedding network
DilatedConv1D.filters = %filters
DilatedConv1D.conv1_kernel_size = 25
DilatedConv1D.n_dil_layers = %n_dil_layers
DilatedConv1D.padding = 'same'
DilatedConv1D.batchnorm = %batchnorm
DilatedConv1D.skip_type = 'residual'

# - Heads
#   - Profile prediction
# This is where we specify parameters/architecture of the head of the SeqModel
ProfileHead.target_name = '{task}/profile'
ProfileHead.net = @DeConv1D()

DeConv1D.n_tasks = %tracks_per_task
DeConv1D.filters = %filters
DeConv1D.tconv_kernel_size = %tconv_kernel_size
DeConv1D.padding = 'same'
DeConv1D.n_hidden = 0
DeConv1D.batchnorm = %batchnorm

ProfileHead.loss = @multinomial_nll
ProfileHead.loss_weight = 1
ProfileHead.postproc_fn = @softmax
ProfileHead.use_bias = %use_bias
ProfileHead.bias_input = 'bias/{task}/profile'
ProfileHead.bias_shape = (None, %n_bias_tracks)
ProfileHead.bias_net = @MovingAverages()
MovingAverages.window_sizes = [1, 50]

#      - evaluate
ProfileHead.metric = @PeakPredictionProfileMetric()
PeakPredictionProfileMetric.pos_min_threshold = 0.015
PeakPredictionProfileMetric.neg_max_threshold = 0.005
PeakPredictionProfileMetric.required_min_pos_counts = 2.5
PeakPredictionProfileMetric.binsizes = [1, 10]
# ---------------------
#   - Total count prediction
ScalarHead.target_name = '{task}/counts'
ScalarHead.net = @GlobalAvgPoolFCN()
GlobalAvgPoolFCN.n_tasks = %tracks_per_task
GlobalAvgPoolFCN.n_splines = 0
GlobalAvgPoolFCN.batchnorm = %batchnorm
ScalarHead.loss = 'mse'
ScalarHead.loss_weight = %lambda
ScalarHead.bias_input = 'bias/{task}/counts'
ScalarHead.use_bias = %use_bias
ScalarHead.bias_shape = (%n_bias_tracks, )
ScalarHead.metric = @RegressionMetrics()

# --------------------------------------------
# training
train.seed = None
train.batch_size = 128
train.epochs = 200
train.early_stop_patience = 5
train.num_workers = 6

# --------------------------------------------
# data
# TODO - shall we try to avoid macros as much as possible?

# seq_width  -> specified from gin-bindings
train.data = @bpnet_data()
bpnet_data.peak_width = %seq_width
bpnet_data.valid_chr = %valid_chr
bpnet_data.test_chr = %test_chr
bpnet_data.include_metadata = False
bpnet_data.tasks = %tasks
bpnet_data.exclude_chr = %exclude_chr
bpnet_data.augment_interval = %augment_interval
bpnet_data.interval_augmentation_shift = 200
bpnet_data.intervals_format = 'bed'

# transform the bias track by aggregating it in a
# sliding window fashion with window sizes of 1 bp (no aggregation) and 50 bp

# TODO - move that into the model -> apply two convolutional layers with constant width

# specified from the CLI
bpnet_data.dataspec = %dataspec
bpnet_data.seq_width = %seq_width
train.train_epoch_frac = 1.0
train.valid_epoch_frac = 1.0

train.eval_report = @report_template()
report_template.name = 'evaluate.ipynb'

# ============================================
# Macros
augment_interval = True

lambda = 10  # count loss weight
filters = 64
tconv_kernel_size = 25
lr = 0.004
n_dil_layers = 9
train.batch_size = 128
batchnorm = False
seq_width = 1000
# TODO - mention the usage of macros

# TODO - important parameters you might want to adjust
valid_chr = ['chr2', 'chr3', 'chr4']
test_chr = ['chr1', 'chr8', 'chr9']
exclude_chr = ['chrX', 'chrY']
# ============================================
# These parameters will be specified from the command line
# (i.e. in `bpnet.cli.train.bpnet_train` function)

# TODO - specified in the training function
# tracks_per_task = 2
# use_bias = True  # set to False if you would like to not use the bias in the model
# n_bias_tracks = 2
# dataspec = 'ChIP-nexus.dataspec.yml'
# tasks = ['Oct4', 'Sox2', 'Nanog', 'Klf4']

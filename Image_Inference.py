import os
import json
import pprint
import datetime
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from bnaf import *
from optim.lr_scheduler import *
import glob
import random
import matplotlib.pyplot as plt

from scipy.optimize import fmin_l_bfgs_b

import functools

def img_heatmap(compute_img_log_p_x, filename, args):
    imgcre = get_image(filename, args)
    # imgcre = np.maximum(imgcre + np.random.uniform(-0.5, 0.5, imgcre.shape), 0)
    rand_box_size = np.int(imgcre.shape[0] * args.rand_box)
    rand_box = np.array([rand_box_size, rand_box_size, 3])
    # rand_box = np.append(tf.cast(tf.multiply(tf.cast(imgcre.shape[:2], tf.float32),tf.constant(0.1)), tf.int32).numpy(), [3])
    rows = imgcre.shape[0]-rand_box[0]
    cols = imgcre.shape[1] - rand_box[1]
    heatmap = np.zeros((np.int(rows/args.spacing), np.int(cols/args.spacing)))
    im_breakup_array = np.zeros((np.int(cols/args.spacing),rand_box_size*rand_box_size*3), dtype=np.float32)
    with tf.device(args.device):
        for i in range(np.int(rows/args.spacing)*args.spacing):
            if not i%args.spacing:
                for j in range(np.int(cols/args.spacing)*args.spacing):
                    if not j%args.spacing:
                        im_breakup_array[np.int(j/args.spacing),:] = tf.reshape((tf.image.crop_to_bounding_box(imgcre, i, j, rand_box_size, rand_box_size)/128 - 1) / args.stdev, [-1]).numpy()
                heatmap[np.int(i/args.spacing), :] = compute_img_log_p_x(x_mb=im_breakup_array).numpy()
    # heatmap = np.zeros((rows, cols))
    # im_breakup_array = np.zeros((cols, rand_box_size*rand_box_size*3), dtype=np.float32)
    # for i in range(rows):
    #     for j in range(cols):
    #         im_breakup_array[j,:] = tf.reshape((tf.image.crop_to_bounding_box(imgcre, i, j, rand_box_size, rand_box_size) - args.mean) / args.stdev, [-1]).numpy()
    #     heatmap[i, :] = compute_img_log_p_x(x_mb=im_breakup_array).numpy()
    return heatmap

def get_image(filename, args):
    img_raw = tf.io.read_file(filename)
    img = tf.image.decode_image(img_raw)
    offset_width = 50
    offset_height = 10
    target_width = 660 - offset_width
    target_height = 470 - offset_height
    imgc = tf.image.crop_to_bounding_box(img, offset_height, offset_width, target_height, target_width)
    # # args.img_size = 0.25;  args.preserve_aspect_ratio = True; args.rand_box = 0.1
    imresize_ = tf.cast(tf.multiply(tf.cast(imgc.shape[:2], tf.float32), tf.constant(args.img_size)), tf.int32)
    return tf.image.resize(imgc, size=imresize_)

def get_dims(filename, args):
    img_raw = tf.io.read_file(filename)
    img = tf.image.decode_image(img_raw)
    offset_width = 50
    offset_height = 10
    target_width = 660 - offset_width
    target_height = 470 - offset_height
    imgc = tf.image.crop_to_bounding_box(img, offset_height, offset_width, target_height, target_width)
    imresize_ = tf.cast(tf.multiply(tf.cast(imgc.shape[:2], tf.float32), tf.constant(args.img_size)), tf.int32)
    imgcre = tf.image.resize(imgc, size=imresize_) / 255
    rand_box_size = np.int(imgcre.shape[0] * args.rand_box)
    rand_box = np.array([rand_box_size, rand_box_size, 3])
    args.n_dims = np.prod(rand_box)


def create_model(args, verbose=False):
    # random.seed(manualSeed)
    # torch.manual_seed(manualSeed)

    tf.random.set_seed(args.manualSeedw)
    np.random.seed(args.manualSeedw)

    dtype_in = tf.float32

    g_constraint = lambda x: tf.nn.relu(x) + 1e-6  ## for batch norm
    flows = []
    for f in range(args.flows):
        # build internal layers for a single flow
        layers = []
        for _ in range(args.layers - 1):
            layers.append(MaskedWeight(args.n_dims * args.hidden_dim,
                                       args.n_dims * args.hidden_dim, dim=args.n_dims, dtype_in=dtype_in))
            layers.append(Tanh(dtype_in=dtype_in))

        flows.append(
            BNAF(layers=[MaskedWeight(args.n_dims, args.n_dims * args.hidden_dim, dim=args.n_dims, dtype_in=dtype_in),
                         Tanh(dtype_in=dtype_in)] + \
                        layers + \
                        [MaskedWeight(args.n_dims * args.hidden_dim, args.n_dims, dim=args.n_dims, dtype_in=dtype_in)], \
                 res=args.residual if f < args.flows - 1 else None, dtype_in=dtype_in
                 )
        )
        ## with batch norm example
        # for _ in range(args.layers - 1):
        #     layers.append(MaskedWeight(args.n_dims * args.hidden_dim,
        #                                args.n_dims * args.hidden_dim, dim=args.n_dims, dtype_in=dtype_in))
        #     layers.append(CustomBatchnorm(gamma_constraint = g_constraint, momentum=args.momentum))
        #     layers.append(Tanh(dtype_in=dtype_in))
        #
        # flows.append(
        #     BNAF(layers = [MaskedWeight(args.n_dims, args.n_dims * args.hidden_dim, dim=args.n_dims, dtype_in=dtype_in), CustomBatchnorm(gamma_constraint = g_constraint, momentum=args.momentum), Tanh(dtype_in=dtype_in)] + \
        #        layers + \
        #        [CustomBatchnorm(scale=False, momentum=args.momentum), MaskedWeight(args.n_dims * args.hidden_dim, args.n_dims, dim=args.n_dims, dtype_in=dtype_in)], \
        #      res=args.residual if f < args.flows - 1 else None, dtype_in= dtype_in
        #      )
        # )

        if f < args.flows - 1:
            flows.append(Permutation(args.n_dims, 'flip'))

        model = Sequential(flows)  # , dtype_in=dtype_in)
        # params = np.sum(np.sum(p.numpy() != 0) if len(p.numpy().shape) > 1 else p.numpy().shape
        #              for p in model.trainable_variables)[0]

    # if verbose:
    #     print('{}'.format(model))
    #     print('Parameters={}, NAF/BNAF={:.2f}/{:.2f}, n_dims={}'.format(params,
    #         NAF_PARAMS[args.dataset][0] / params, NAF_PARAMS[args.dataset][1] / params, args.n_dims))

    # if args.save and not args.load:
    #     with open(os.path.join(args.load or args.path, 'results.txt'), 'a') as f:
    #         print('Parameters={}, NAF/BNAF={:.2f}/{:.2f}, n_dims={}'.format(params,
    #             NAF_PARAMS[args.dataset][0] / params, NAF_PARAMS[args.dataset][1] / params, args.n_dims), file=f)

    return model


def load_model(args, root, load_start_epoch=False):
    # def f():
    print('Loading model..')
    root.restore(tf.train.latest_checkpoint(args.load or args.path))
    # root.restore(os.path.join(args.load or args.path, 'checkpoint'))
    # if load_start_epoch:
    #     args.start_epoch = tf.train.get_global_step().numpy()
    # return f

# @tf.function
def compute_log_p_x(model, x_mb):
    ## use tf.gradient + tf.convert_to_tensor + tf.GradientTape(persistent=True) to clean up garbage implementation in bnaf.py
    y_mb, log_diag_j_mb = model(x_mb)
    log_p_y_mb = tf.reduce_sum(tfp.distributions.Normal(tf.zeros_like(y_mb), tf.ones_like(y_mb)).log_prob(y_mb),
                               axis=-1)  # .sum(-1)
    return log_p_y_mb + log_diag_j_mb


class parser_:
    pass

def main():
    # config = tf.compat.v1.ConfigProto()
    # config.gpu_options.allow_growth = True
    # config.log_device_placement = True
    # tf.compat.v1.enable_eager_execution(config=config)

    # tf.config.experimental_run_functions_eagerly(True)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    args = parser_()
    args.device = '/gpu:0'  # '/gpu:0'
    args.dataset = 'corn'  # 'gq_ms_wheat_johnson'#'gq_ms_wheat_johnson' #['gas', 'bsds300', 'hepmass', 'miniboone', 'power']
    args.learning_rate = np.float32(1e-2)
    args.batch_dim = 50
    args.clip_norm = 0.1
    args.epochs = 5000
    args.patience = 10
    args.cooldown = 10
    args.decay = 0.5
    args.min_lr = 5e-4
    args.flows = 6
    args.layers = 1
    args.hidden_dim = 12
    args.residual = 'gated'
    args.expname = ''
    args.load = r'C:\Users\justjo\PycharmProjects\BNAF_tensorflow_eager\checkpoint\corn_layers1_h12_flows6_resize0.25_boxsize0.1_gated_2019-08-25-22-18-30'
    args.save = True
    args.tensorboard = 'tensorboard'
    args.early_stopping = 15
    args.maxiter = 5000
    args.factr = 1E1
    args.regL2 = -1
    args.regL1 = -1
    args.manualSeed = None
    args.manualSeedw = None
    args.momentum = 0.9  ## batch norm momentum
    args.prefetch_size = 10  # data pipeline prefetch buffer size
    args.parallel = 16  # data pipeline parallel processes
    args.img_size = 0.25;  ## resize img between 0 and 1
    args.preserve_aspect_ratio = True;  ##when resizing
    args.rand_box = 0.1  ##relative size of random box from image
    args.spacing = 1

    args.path = os.path.join('checkpoint', '{}{}_layers{}_h{}_flows{}_resize{}_boxsize{}{}_{}'.format(
        args.expname + ('_' if args.expname != '' else ''),
        args.dataset, args.layers, args.hidden_dim, args.flows, args.img_size, args.rand_box, '_' + args.residual if args.residual else '',
        str(datetime.datetime.now())[:-7].replace(' ', '-').replace(':', '-')))

    print('Loading dataset..')

    fnames = glob.glob('data/GQ_Images/*.png')

    ##set n_dims
    get_dims(fnames[0], args)

    if args.save and not args.load:
        print('Creating directory experiment..')
        os.mkdir(args.path)
        with open(os.path.join(args.path, 'args.json'), 'w') as f:
            json.dump(str(args.__dict__), f, indent=4, sort_keys=True)

    print('Creating BNAF model..')
    with tf.device(args.device):
        model = create_model(args, verbose=True)

    ### debug
    # data_loader_train_ = tf.contrib.eager.Iterator(data_loader_train)
    # x = data_loader_train_.get_next()
    # a = model(x)

    ## tensorboard and saving
    writer = tf.summary.create_file_writer(os.path.join(args.tensorboard, args.load or args.path))
    writer.set_as_default()
    tf.compat.v1.train.get_or_create_global_step()

    global_step = tf.compat.v1.train.get_global_step()
    global_step.assign(0)

    root = None
    args.start_epoch = 0

    print('Creating optimizer..')
    with tf.device(args.device):
        optimizer = tf.optimizers.Adam()
    root = tf.train.Checkpoint(optimizer=optimizer,
                               model=model,
                               optimizer_step=tf.compat.v1.train.get_global_step())

    if args.load:
        load_model(args, root, load_start_epoch=True)

    print('Creating scheduler..')
    # use baseline to avoid saving early on
    scheduler = EarlyStopping(model=model, patience=args.early_stopping, args=args, root=root)

    heat_map_func = functools.partial(compute_log_p_x, model=model)

    heat_map = []
    fnames = glob.glob('data/GQ_Images/test_images_broken/*.png')
    fnames = glob.glob('data/GQ_Images/*.png')

    heat_map.extend(img_heatmap(heat_map_func, f, args) for f in fnames)
    heatmap_ = np.array(heat_map)

    ## johnsonsu xfrm for density fit (5.4155884341570175, 4.78009012658631, 622.0617883438022, 214.5187927541507)
    # dist = [5.4155884341570175, 4.78009012658631, 622.0617883438022, 214.5187927541507]
    dist = (8.144493590964167, 6.017963993607797, 740.3910154966748, 219.38576508100834)
    heatmap_ = (np.arcsinh((heatmap_ - dist[-2]) / dist[-1]) * dist[1] + dist[0])



    heatmap_ = tf.sigmoid((np.arcsinh((heatmap_ - dist[-2]) / dist[-1]) * dist[1] + dist[0])).numpy()
    ##call function directly first
    i = 0
    plt.figure();plt.imshow(get_image(fnames[i], args)/255)
    plt.figure()
    plt.imshow(heatmap_[i], cmap='hot', interpolation='nearest', vmin=0, vmax=1, alpha=1)

if __name__ == '__main__':
    main()

##"C:\Program Files\Git\bin\sh.exe" --login -i

#### tensorboard --logdir=C:\Users\justjo\PycharmProjects\BNAF_tensorflow_eager\tensorboard\checkpoint
## http://localhost:6006/


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
import struct
import gzip
import pathlib
import scipy
import sklearn.mixture
# from scipy.optimize import fmin_l_bfgs_b

import functools

tf.random.set_seed(None)

def img_preprocessing(imgcre, args):
    # rand_box = np.append(tf.cast(tf.multiply(tf.cast(imgcre.shape[:2], tf.float32),tf.constant(0.1)), tf.int32).numpy(), [3])
    rand_crop = tf.image.random_crop(imgcre, args.rand_box)
    rand_crop = tf.minimum(tf.nn.relu(rand_crop + tf.random.uniform(rand_crop.shape, -0.5, 0.5)), 255) ## dequantize
    if type(args.vh) is np.ndarray:
        return tf.squeeze(tf.matmul(tf.reshape(rand_crop/128 - 1, [1,-1]), args.vh.T))
    else:
        return tf.reshape(rand_crop/128 - 1, [-1])

def img_load(filename, args):
    img_raw = tf.io.read_file(filename)
    img = tf.image.decode_image(img_raw)
    offset_width = 50
    offset_height = 10
    target_width = 660 - offset_width
    target_height = 470 - offset_height
    imgc = tf.image.crop_to_bounding_box(img, offset_height, offset_width, target_height, target_width)
    # # args.img_size = 0.25;  args.preserve_aspect_ratio = True; args.rand_box = 0.1
    imresize_ = tf.cast(tf.multiply(tf.cast(imgc.shape[:2], tf.float32), tf.constant(args.img_size)), tf.int32)
    imgcre = tf.image.resize(imgc, size=imresize_)
    return imgcre

def dequantize(img):
    return img + tf.random.uniform(img.shape, -0.00390625, 0.00390625) ## 0.5/128

def load_dataset(args):

    tf.random.set_seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    random.seed(args.manualSeed)

    trainval = glob.glob(r'D:\GQC_Images\GQ_Images\Corn_2017_2018/*.png')
    train_data = np.vstack([np.expand_dims(img_load(x,args),axis=0) for x in trainval])
    cont_data = glob.glob(r'D:\GQC_Images\GQ_Images\test_images_broken/*.png')
    cont_data = np.vstack([np.expand_dims(img_load(x,args),axis=0) for x in cont_data])

    args.rand_box_size = np.int(train_data[0].shape[0] * args.rand_box_init)
    args.rand_box = np.array([args.rand_box_size, args.rand_box_size, 3])
    args.n_dims = np.prod(args.rand_box)

    if args.vh:
        cliplist = []
        for n in range(10):
            cliplist.append(np.vstack([img_preprocessing(x, args) for x in train_data]))
        svdmat = np.vstack(cliplist)
        _, _, args.vh = scipy.linalg.svd(svdmat, full_matrices=False)

    img_preprocessing_ = functools.partial(img_preprocessing, args=args)
    # img_preprocessing_ = dequantize

    dataset_train = tf.data.Dataset.from_tensor_slices(train_data)#.float().to(args.device)
    dataset_train = dataset_train.shuffle(buffer_size=len(train_data)).map(img_preprocessing_, num_parallel_calls=args.parallel).batch(batch_size=args.batch_dim).prefetch(buffer_size=args.prefetch_size)
    # dataset_train = dataset_train.shuffle(buffer_size=len(train)).batch(batch_size=args.batch_dim).prefetch(buffer_size=args.prefetch_size)

    dataset_valid = tf.data.Dataset.from_tensor_slices(train_data)#.float().to(args.device)
    dataset_valid = dataset_valid.map(img_preprocessing_, num_parallel_calls=args.parallel).batch(batch_size=args.batch_dim*2).prefetch(buffer_size=args.prefetch_size)
    # dataset_valid = dataset_valid.batch(batch_size=args.batch_dim*2).prefetch(buffer_size=args.prefetch_size)

    dataset_test = tf.data.Dataset.from_tensor_slices(train_data)#.float().to(args.device)
    dataset_test = dataset_test.map(img_preprocessing_, num_parallel_calls=args.parallel).batch(batch_size=args.batch_dim*2).prefetch(buffer_size=args.prefetch_size)

    dataset_cont = tf.data.Dataset.from_tensor_slices(cont_data)#.float().to(args.device)
    dataset_cont = dataset_cont.map(img_preprocessing_, num_parallel_calls=args.parallel).batch(batch_size=args.batch_dim*2).prefetch(buffer_size=args.prefetch_size)

    args.n_dims = img_preprocessing_(train_data[0]).shape[0]

    # args.n_dims = train.shape[1]
    return dataset_train, dataset_valid, dataset_test, dataset_cont

def create_model(args, verbose=False):

    # random.seed(manualSeed)
    # torch.manual_seed(manualSeed)

    tf.random.set_seed(args.manualSeedw)
    np.random.seed(args.manualSeedw)

    dtype_in = tf.float32

    g_constraint = lambda x: tf.nn.relu(x) + 1e-6 ## for batch norm
    flows = []
    for f in range(args.flows):
        #build internal layers for a single flow
        layers = []
        for _ in range(args.layers - 1):
            layers.append(MaskedWeight(args.n_dims * args.hidden_dim,
                                       args.n_dims * args.hidden_dim, dim=args.n_dims, dtype_in=dtype_in))
            layers.append(Tanh(dtype_in=dtype_in))

        flows.append(
            BNAF(layers = [MaskedWeight(args.n_dims, args.n_dims * args.hidden_dim, dim=args.n_dims, dtype_in=dtype_in), Tanh(dtype_in=dtype_in)] + \
               layers + \
               [MaskedWeight(args.n_dims * args.hidden_dim, args.n_dims, dim=args.n_dims, dtype_in=dtype_in)], \
             res=args.residual if f < args.flows - 1 else None, dtype_in= dtype_in
             )
        )
        ## with batch norm example
 #        for _ in range(args.layers - 1):
 #            layers.append(MaskedWeight(args.n_dims * args.hidden_dim,
 #                                       args.n_dims * args.hidden_dim, dim=args.n_dims, dtype_in=dtype_in))
 #            layers.append(CustomBatchnorm(gamma_constraint = g_constraint, momentum=args.momentum, renorm=True, renorm_momentum=0.9))
 #            layers.append(Tanh(dtype_in=dtype_in))
 #
 #        flows.append(
 #            BNAF(layers = [MaskedWeight(args.n_dims, args.n_dims * args.hidden_dim, dim=args.n_dims, dtype_in=dtype_in), CustomBatchnorm(gamma_constraint = g_constraint, momentum=args.momentum, renorm=True, renorm_momentum=0.9), Tanh(dtype_in=dtype_in)] + \
 #               layers + \
 #               # [CustomBatchnorm(scale=False, momentum=args.momentum), MaskedWeight(args.n_dims * args.hidden_dim, args.n_dims, dim=args.n_dims, dtype_in=dtype_in)], \
 #                 [MaskedWeight(args.n_dims * args.hidden_dim, args.n_dims, dim=args.n_dims, dtype_in=dtype_in)], \
 # \
 #                 res=args.residual if f < args.flows - 1 else None, dtype_in= dtype_in
 #             )
 #        )

        if f < args.flows - 1:
            flows.append(Permutation(args.n_dims, 'flip'))

        model = Sequential(flows)#, dtype_in=dtype_in)
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
# def compute_log_p_x(model, x_mb):
#     ## use tf.gradient + tf.convert_to_tensor + tf.GradientTape(persistent=True) to clean up garbage implementation in bnaf.py
#     y_mb, log_diag_j_mb = model(x_mb)
#     log_p_y_mb = tf.reduce_sum(tfp.distributions.Normal(tf.zeros_like(y_mb), tf.ones_like(y_mb)).log_prob(y_mb), axis=-1)#.sum(-1)
#     return log_p_y_mb + log_diag_j_mb

## batch norm
def compute_log_p_x(model, x_mb, training=False):
    ## use tf.gradient + tf.convert_to_tensor + tf.GradientTape(persistent=True) to clean up garbage implementation in bnaf.py
    y_mb, log_diag_j_mb = model(x_mb, training=training)
    log_p_y_mb = tf.reduce_sum(tfp.distributions.Normal(tf.zeros_like(y_mb), tf.ones_like(y_mb)).log_prob(y_mb), axis=-1)#.sum(-1)
    return log_p_y_mb + log_diag_j_mb

# @tf.function
def train(model, optimizer, scheduler, data_loader_train, data_loader_valid, data_loader_test, data_loader_cont, args):
    
    epoch = args.start_epoch
    for epoch in range(args.start_epoch, args.start_epoch + args.epochs):

        # t = tqdm(data_loader_train, smoothing=0, ncols=80)
        train_loss = []
        
        for x_mb in data_loader_train:
            with tf.GradientTape() as tape:
                loss = - tf.reduce_mean(compute_log_p_x(model, x_mb, training=True)) #negative -> minimize to maximize liklihood

            grads = tape.gradient(loss, model.trainable_variables)
            grads = [None if grad is None else tf.clip_by_norm(grad, clip_norm=args.clip_norm) for grad in grads]
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            train_loss.append(loss)

            tf.compat.v1.train.get_global_step().assign_add(1)

        ## potentially update batch norm variables manuallu
        ## variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='batch_normalization')

        train_loss = np.mean(train_loss)
        validation_loss = -tf.reduce_mean([tf.reduce_mean(compute_log_p_x(model, x_mb, training=False)) for x_mb in data_loader_valid])
        cont_loss = tf.reduce_mean([tf.reduce_mean(compute_log_p_x(model, x_mb, training=False)) for x_mb in data_loader_cont])

        # print('Epoch {:3}/{:3} -- train_loss: {:4.3f} -- validation_loss: {:4.3f}'.format(
        #     epoch + 1, args.start_epoch + args.epochs, train_loss, validation_loss))


        stop = scheduler.on_epoch_end(epoch = epoch, monitor=validation_loss)

        if args.tensorboard:
            # with tf.contrib.summary.always_record_summaries():
                tf.summary.scalar('loss/validation', validation_loss,tf.compat.v1.train.get_global_step())
                tf.summary.scalar('loss/train', train_loss, tf.compat.v1.train.get_global_step())
                tf.summary.scalar('loss/cont', cont_loss, tf.compat.v1.train.get_global_step())
                # writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch + 1)
                # writer.add_scalar('loss/validation', validation_loss.item(), epoch + 1)
                # writer.add_scalar('loss/train', train_loss.item(), epoch + 1)

        if stop:
            break

    # validation_loss = - tf.reduce_mean([tf.reduce_mean(compute_log_p_x(model, x_mb)) for x_mb in data_loader_valid])
    # test_loss = - tf.reduce_mean([tf.reduce_mean(compute_log_p_x(model, x_mb)) for x_mb in data_loader_test])
    #
    # print('###### Stop training after {} epochs!'.format(epoch + 1))
    # print('Validation loss: {:4.3f}'.format(validation_loss))
    # print('Test loss:       {:4.3f}'.format(test_loss))
    # print('Contrastive loss:       {:4.3f}'.format(cont_loss))
    
    # if args.save:
    #     with open(os.path.join(args.load or args.path, 'results.txt'), 'a') as f:
    #         print('###### Stop training after {} epochs!'.format(epoch + 1), file=f)
    #         print('Validation loss: {:4.3f}'.format(validation_loss), file=f)
    #         print('Test loss:       {:4.3f}'.format(test_loss), file=f)

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
    args.dataset = 'corn' #'gq_ms_wheat_johnson'#'gq_ms_wheat_johnson' #['gas', 'bsds300', 'hepmass', 'miniboone', 'power']
    args.learning_rate = np.float32(1e-2)
    args.batch_dim = 500
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
    args.load = ''#r'C:\Users\justjo\PycharmProjects\BNAF_tensorflow_eager\checkpoint\corn_layers1_h12_flows6_resize0.25_boxsize0.1_gated_2019-08-24-11-07-09'
    args.save = True
    args.tensorboard = r'D:\pycharm_projects\GQC_images_tensorboard'
    args.early_stopping = 30
    args.regL2 = -1
    args.regL1 = -1
    args.manualSeed = None
    args.manualSeedw = None
    args.momentum = 0.25 ## batch norm momentum
    args.prefetch_size = 1 #data pipeline prefetch buffer size
    args.parallel = 16 #data pipeline parallel processes
    args.img_size = 0.25; ## resize img between 0 and 1
    args.preserve_aspect_ratio = True; ##when resizing
    args.rand_box = 0.1 ##relative size of random box from image
    args.p_val = 0.2
    args.vh = 1 #0 =no, 1=yes

    args.path = os.path.join(r'D:\pycharm_projects\GQC_images_tensorboard', '{}{}_layers{}_h{}_flows{}_resize{}_boxsize{}{}_{}'.format(
        args.expname + ('_' if args.expname != '' else ''),
        args.dataset, args.layers, args.hidden_dim, args.flows, args.img_size, args.rand_box, '_' + args.residual if args.residual else '',
        str(datetime.datetime.now())[:-7].replace(' ', '-').replace(':', '-')))

    print('Loading dataset..')

    data_loader_train, data_loader_valid, data_loader_test, data_loader_cont = load_dataset(args)

    if args.save and not args.load:
        print('Creating directory experiment..')
        pathlib.Path(args.path).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(args.path, 'args.json'), 'w') as f:
            json.dump(str(args.__dict__), f, indent=4, sort_keys=True)

    # pathlib.Path(args.tensorboard).mkdir(parents=True, exist_ok=True)

    print('Creating BNAF model..')
    with tf.device(args.device):
        model = create_model(args, verbose=True)

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
    scheduler = EarlyStopping(model=model, patience=args.early_stopping, args = args, root = root)

    with tf.device(args.device):
        train(model, optimizer, scheduler, data_loader_train, data_loader_valid, data_loader_test, data_loader_cont, args)

    ## save??
    scheduler.save_model()
    if type(args.vh) is np.ndarray:
        np.save(args.path + '/vh.npy', args.vh)

if __name__ == '__main__':
    main()

##"C:\Program Files\Git\bin\sh.exe" --login -i

#### tensorboard --logdir=D:\pycharm_projects\GQC_images_tensorboard
## http://localhost:6006/


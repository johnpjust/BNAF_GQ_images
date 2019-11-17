import tensorflow as tf
import math
import numpy as np
import tensorflow_probability as tfp


class Sequential(tf.keras.models.Sequential):
    """
    Class that extends ``torch.nn.Sequential`` for computing the output of
    the function alongside with the log-det-Jacobian of such transformation.
    """

    # def __init__(self, layers=None, name=None)#, dtype_in = tf.float32):
    #     super(Sequential, self).__init__(name=name)
    #     self.supports_masking = True
    #     self._build_input_shape = None
    #     self._compute_output_and_mask_jointly = True
    #
    #     self._layer_call_argspecs = {}
    #     self.dtype_in = dtype_in
    #     # Add to the model any layers passed to the constructor.
    #     if layers:
    #         for layer in layers:
    #             self.add(layer)

    # def call(self, inputs: tf.Tensor):
    @tf.function
    def call(self, inputs, training=None, mask=None):
        """
        Parameters
        ----------
        inputs : ``torch.Tensor``, required.
            The input tensor.
        Returns
        -------
        The output tensor and the log-det-Jacobian of this transformation.
        """

        log_det_jacobian = tf.cast(0., tf.float32)
        # log_det_jacobian = 0.
        # for i, module in enumerate(self._modules.values()):
        for i, layer in enumerate(self.layers):
            inputs, log_det_jacobian_ = layer(inputs, training=training)
            log_det_jacobian = log_det_jacobian + log_det_jacobian_
        return inputs, log_det_jacobian


class BNAF(tf.keras.models.Sequential):
    """
    Class that extends ``torch.nn.Sequential`` for constructing a Block Neural
    Normalizing Flow.
    """

    def __init__(self, layers=None, name=None, res: str = None, dtype_in=tf.float32):
        # def __init__(self, *args, res: str = None):
        """
        Parameters
        ----------
        *args : ``Iterable[torch.nn.Module]``, required.
            The modules to use.
        res : ``str``, optional (default = None).
            Which kind of residual connection to use. ``res = None`` is no residual
            connection, ``res = 'normal'`` is ``x + f(x)`` and ``res = 'gated'`` is
            ``a * x + (1 - a) * f(x)`` where ``a`` is a learnable parameter.
        """

        # super(BNAF, self).__init__(*args)
        super(BNAF, self).__init__(name=name)
        self.supports_masking = True
        self._build_input_shape = None
        self._compute_output_and_mask_jointly = True

        # Add to the model any layers passed to the constructor.
        if layers:
            for layer in layers:
                self.add(layer)

        self.res = res

        if res == 'gated':
            initializer = tf.random_normal_initializer()
            self.gate = tf.Variable(name='gate', initial_value=tf.cast(initializer(shape=(1,)), dtype_in))
            # self.gate = torch.nn.Parameter(torch.nn.init.normal_(torch.Tensor(1)))

    # def forward(self, inputs : tf.Tensor):
    # def call(self, inputs: tf.Tensor):
    @tf.function
    def call(self, inputs, training=None, mask=None):

        """
        Parameters
        ----------
        inputs : ``torch.Tensor``, required.
            The input tensor.
        Returns
        -------
        The output tensor and the log-det-Jacobian of this transformation.
        """
        outputs = inputs
        grad = None

        ### apply layers in TF and get gradients...module in pytorch == layers in tf
        # for module in self._modules.values(): #pytorch implementation
        for layer in self.layers:
            outputs, grad = layer(outputs, grad, training=training)  # not sure if use "layer" or "layer.call"
            grad = grad if len(grad.shape) == 4 else tf.reshape(grad, (grad.shape + [1, 1]))

        # return outputs, grad ## debug

        assert inputs.shape[-1] == outputs.shape[-1]
        grad = tf.squeeze(grad)
        reduce_sum = len(grad.shape) > 1

        if reduce_sum:
            if self.res == 'normal':
                return inputs + outputs, tf.reduce_sum(tf.keras.activations.softplus(tf.squeeze(grad)), axis=-1)
            elif self.res == 'gated':
                return tf.nn.sigmoid(self.gate) * outputs + (1 - tf.nn.sigmoid(self.gate)) * inputs, \
                       tf.reduce_sum(tf.nn.softplus(tf.squeeze(grad) + self.gate) - \
                                     tf.nn.softplus(self.gate), axis=-1)
            else:
                return outputs, tf.reduce_sum(tf.squeeze(grad), axis=-1)
        else:
            if self.res == 'normal':
                return inputs + outputs, tf.keras.activations.softplus(grad)
            elif self.res == 'gated':
                return tf.nn.sigmoid(self.gate) * outputs + (1 - tf.nn.sigmoid(self.gate)) * inputs, \
                       tf.nn.softplus(grad + self.gate) - tf.nn.softplus(self.gate)
            else:
                return outputs, grad

    def _get_name(self):
        return 'BNAF(res={})'.format(self.res)


# class Permutation(torch.nn.Module):
class Permutation(tf.keras.layers.Layer):
    """
    Module that outputs a permutation of its input.
    """

    def __init__(self, in_features: int, p: list = None):
        """
        Parameters
        ----------
        in_features : ``int``, required.
            The number of input features.
        p : ``list`` or ``str``, optional (default = None)
            The list of indices that indicate the permutation. When ``p`` is not a
            list, if ``p = 'flip'``the tensor is reversed, if ``p = None`` a random
            permutation is applied.
        """

        super(Permutation, self).__init__()

        self.in_features = in_features

        if p is None:
            self.p = tfp.bijectors.Permute(np.random.permutation(in_features))
        elif p == 'flip':
            self.p = tfp.bijectors.Permute(list(reversed(range(in_features))))
        else:
            self.p = tfp.bijectors.Permute(p)

    @tf.function
    def call(self, inputs: tf.Tensor, **kwargs):
        """
        Parameters
        ----------
        inputs : ``torch.Tensor``, required.
            The input tensor.
        Returns
        -------
        The permuted tensor and the log-det-Jacobian of this permutation.
        """

        # return inputs[:,self.p], 0
        return self.p.forward(inputs), 0.

    def __repr__(self):
        return 'Permutation(in_features={}, p={})'.format(self.in_features, self.p)


# class MaskedWeight(torch.nn.Module):
class MaskedWeight(tf.keras.layers.Layer):
    """
    Module that implements a linear layer with block matrices with positive diagonal blocks.
    Moreover, it uses Weight Normalization (https://arxiv.org/abs/1602.07868) for stability.
    """

    def __init__(self, in_features: int, out_features: int, dim: int, bias: bool = True, dtype_in=tf.float32):
        """
        Parameters
        ----------
        in_features : ``int``, required.
            The number of input features per each dimension ``dim``.
        out_features : ``int``, required.
            The number of output features per each dimension ``dim``.
        dim : ``int``, required.
            The number of dimensions of the input of the flow.
        bias : ``bool``, optional (default = True).
            Whether to add a parametrizable bias.
        """

        super(MaskedWeight, self).__init__()
        self.in_features, self.out_features, self.dim = in_features, out_features, dim
        self.dtype_in = dtype_in
        if self.dtype_in == tf.float32:
            self.dtype_in_np = np.float32
        else:
            self.dtype_in_np = np.float64

        weight = np.zeros((out_features, in_features))

        ## tensorflow init
        initializer = tf.initializers.GlorotUniform()
        for i in range(dim):
            weight[(i * out_features // dim):((i + 1) * out_features // dim), 0:((i + 1) * in_features // dim)] = \
                tf.Variable(name="w", initial_value=tf.cast(
                    initializer(shape=[out_features // dim, (i + 1) * in_features // dim]), self.dtype_in),
                            dtype=self.dtype_in, trainable=False).numpy()
        # ## torch init
        # for i in range(dim):
        #     weight[(i * out_features // dim):((i + 1) * out_features // dim), 0:((i + 1) * in_features // dim)] = torch.nn.init.xavier_uniform_(
        #         torch.Tensor(out_features // dim, (i + 1) * in_features // dim)).numpy()

        # with tf.variable_scope("params", reuse=False):
        self._weight = tf.Variable(name="off_diagonal", initial_value=tf.cast(weight, dtype=self.dtype_in),
                                   dtype=self.dtype_in)
        ## tf init
        self._diag_weight = tf.Variable(name="diag",
                                        initial_value=np.log(np.random.uniform(0, 1, size=(out_features, 1))).astype(
                                            self.dtype_in_np),
                                        dtype=self.dtype_in)  # maybe takes log because we're going to take exp later?
        self.bias = tf.Variable(name="bias", initial_value=tf.cast(
            tf.random.uniform(shape=(out_features,), minval=-1 / math.sqrt(out_features),
                              maxval=1 / math.sqrt(out_features)), self.dtype_in)) if bias else tf.cast(0,
                                                                                                        self.dtype_in)
        # ## torch init
        # self._diag_weight = tf.get_variable("diag", initializer=torch.nn.init.uniform_(torch.Tensor(out_features, 1)).log().numpy(), dtype=tf.float32) #maybe takes log because we're going to take exp later?
        # self.bias = tf.get_variable("bias", initializer=torch.nn.init.uniform_(torch.Tensor(out_features),
        #                        -1 / math.sqrt(out_features),
        #                        1 / math.sqrt(out_features)).numpy()) if bias else 0

        mask_d = np.zeros_like(weight)
        for i in range(dim):
            mask_d[i * (out_features // dim):(i + 1) * (out_features // dim),
            i * (in_features // dim):(i + 1) * (in_features // dim)] = 1

        # self.register_buffer('mask_d', mask_d)
        self.mask_d = tf.constant(name='mask_d', value=mask_d, dtype=self.dtype_in)

        mask_o = np.ones_like(weight, dtype=self.dtype_in_np)
        for i in range(dim):
            mask_o[i * (out_features // dim):(i + 1) * (out_features // dim),
            i * (in_features // dim):] = 0

        # self.register_buffer('mask_o', mask_o)
        self.mask_o = tf.constant(name='mask_o', value=mask_o, dtype=self.dtype_in)

    def get_weights(self):
        """
        Computes the weight matrix using masks and weight normalization.
        It also compute the log diagonal blocks of it.
        """

        # error in original here i think -- should be self._diag_weight or w_squared_norm is not correct
        w = tf.multiply(tf.exp(self._weight), self.mask_d) + tf.multiply(self._weight, self.mask_o)
        # w = tfp.bijectors.transform_diagonal(self._weight)
        # w = tf.multiply(tf.exp(self._diag_weight), self.mask_d) + tf.multiply(self._weight, self.mask_o)

        w_squared_norm = tf.reduce_sum(tf.math.square(w), axis=-1, keepdims=True)

        w = tf.exp(self._diag_weight) * w / tf.sqrt(w_squared_norm)

        ## this piece feeds the log-determinant of the jacobian -- the diagonals are all that are needed
        # and they are extracted with the boolean_mask in the return argument below
        wpl = self._diag_weight + self._weight - 0.5 * tf.math.log(w_squared_norm)

        # return tf.transpose(w), tf.transpose(wpl)[self.mask_d.byte().t()].view(
        #     self.dim, self.in_features // self.dim, self.out_features // self.dim)

        return tf.transpose(w), tf.reshape(
            tf.boolean_mask(tf.transpose(wpl), tf.transpose(tf.cast(self.mask_d, tf.bool))), (
                self.dim, self.in_features // self.dim, self.out_features // self.dim))

    # def get_weights(self): ## no weight norm
    #
    #     w = tf.multiply(tf.exp(self._weight), self.mask_d) + tf.multiply(self._weight, self.mask_o)
    #     wpl = self._weight
    #
    #     return tf.transpose(w), tf.reshape(
    #         tf.boolean_mask(tf.transpose(wpl), tf.transpose(tf.cast(self.mask_d, tf.bool))), (
    #             self.dim, self.in_features // self.dim, self.out_features // self.dim))

    # def forward(self, inputs, grad : torch.Tensor = None):
    @tf.function
    def call(self, inputs: tf.Tensor, grad: tf.Tensor = None, **kwargs):
        """
        Parameters
        ----------
        inputs : ``torch.Tensor``, required.
            The input tensor.
        grad : ``torch.Tensor``, optional (default = None).
            The log diagonal block of the partial Jacobian of previous transformations.
        Returns
        -------
        The output tensor and the log diagonal blocks of the partial log-Jacobian of previous
        transformations combined with this transformation.
        """

        w, wpl = self.get_weights()

        # g = wpl.transpose(-2, -1).unsqueeze(0).repeat(inputs.shape[0], 1, 1, 1)
        grad_perm = list(range(len(wpl.shape)))
        grad_perm[-1] = len(grad_perm) - 2
        grad_perm[-2] = len(grad_perm) - 1
        g = tf.tile(tf.expand_dims(tf.transpose(wpl, perm=grad_perm), axis=0), (inputs.shape[0], 1, 1, 1))

        # return inputs.matmul(w) + self.bias, torch.logsumexp(
        #     g.unsqueeze(-2) + grad.transpose(-2, -1).unsqueeze(-3), -1) if grad is not None else g

        if grad is not None:
            grad_perm = list(range(len(grad.shape)))
            grad_perm[-1] = len(grad_perm) - 2
            grad_perm[-2] = len(grad_perm) - 1

        return tf.matmul(inputs, w) + self.bias, tf.reduce_logsumexp(
            tf.expand_dims(g, axis=-2) + tf.expand_dims(tf.transpose(grad, perm=grad_perm), axis=-3),
            axis=-1) if grad is not None else g

    def __repr__(self):
        return 'MaskedWeight(in_features={}, out_features={}, dim={}, bias={})'.format(
            self.in_features, self.out_features, self.dim, not isinstance(self.bias, int))


class Tanh(tf.keras.layers.Layer):
    """
    Class that extends ``torch.nn.Tanh`` additionally computing the log diagonal
    blocks of the Jacobian.
    """

    def __init__(self, dtype_in=tf.float32):
        super(Tanh, self).__init__()

        self.dtype_in = dtype_in

    @tf.function
    def call(self, inputs, grad: tf.Tensor = None, **kwargs):
        """
        Parameters
        ----------
        inputs : ``torch.Tensor``, required.
            The input tensor.
        grad : ``torch.Tensor``, optional (default = None).
            The log diagonal blocks of the partial Jacobian of previous transformations.
        Returns
        -------
        The output tensor and the log diagonal blocks of the partial log-Jacobian of previous 
        transformations combined with this transformation.
        """
        # g = - 2 * (inputs - tf.math.log(2.) + tf.keras.activations.softplus(- 2. * inputs))
        # return tf.tanh(inputs), (tf.reshape(g,grad.shape) + grad) if grad is not None else g

        g = - 2 * tf.add(tf.subtract(inputs, tf.cast(tf.math.log(2.), self.dtype_in)),
                         tf.keras.activations.softplus(- 2. * inputs))
        return tf.tanh(inputs), tf.add(tf.reshape(g, grad.shape), grad) if grad is not None else g


class CustomBatchnorm(tf.keras.layers.BatchNormalization):
    ##gamma_constraint = lambda x: tf.exp(x) + 1e-6
    ##gamma_constraint = lambda x: tf.nn.relu(x) + 1e-6
    @tf.function
    def call(self, inputs, grad, training=None):
        normed_vars = super().call(inputs, training)
        g = self._inverse_log_det_jacobian(inputs, not training)
        if grad is not None:
            bn_g = tf.reshape(g, grad.shape[1:])
            zs = tf.zeros([g.shape[0], *bn_g.shape], dtype=tf.float32) + tf.expand_dims(bn_g, 0)
            g = tf.add(zs, grad)
        return normed_vars, g

    ##tfp.bijectors.batch_normalization()
    def _inverse_log_det_jacobian(self, y, use_saved_statistics=False):
        if not self.built:
            # Create variables.
            self.build(y.shape)

        event_dims = self.axis
        reduction_axes = [i for i in range(len(y.shape)) if i not in event_dims]

        # At training-time, ildj is computed from the mean and log-variance across
        # the current minibatch.
        # We use multiplication instead of tf.where() to get easier broadcasting.
        log_variance = tf.math.log(
            tf.where(use_saved_statistics,
                     self.moving_variance,
                     tf.nn.moments(x=y, axes=reduction_axes, keepdims=True)[1]) +
            self.epsilon)

        # to happen across all axes.
        # `gamma` and `log Var(y)` reductions over event_dims.
        # Log(total change in area from gamma term).
        log_gamma = tf.math.log(self.gamma) if self.gamma is not None else 0
        log_total_gamma = tf.reduce_sum(log_gamma)

        # Log(total change in area from log-variance term).
        log_total_variance = tf.reduce_sum(log_variance)
        # The ildj is scalar, as it does not depend on the values of x and are
        # constant across minibatch elements.

        ## by appendix B of https://arxiv.org/pdf/1705.07057.pdf the gamma should be exponentiated
        ## hence the gamma contraint is already tf.exp(x) + epsilon
        return log_gamma - 0.5 * log_variance
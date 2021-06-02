import json
import mala
import tensorflow as tf
from mala.networks.unet import crop_zyx

def create_auto(input_shape, output_shape, name):

    tf.reset_default_graph()

    with tf.variable_scope('lsd'):

        raw = tf.placeholder(tf.float32, shape=input_shape)
        raw_batched = tf.reshape(raw, (1, 1) + input_shape)

        unet, _, _ = mala.networks.unet(
                raw_batched,
                12,
                6,
                [[2,2,2],[2,2,2],[3,3,3]])

        embedding_batched, _ = mala.networks.conv_pass(
            unet,
            kernel_sizes=[1],
            num_fmaps=10,
            activation='sigmoid',
            name='embedding')

        embedding_batched = crop_zyx(embedding_batched, (1, 10) + output_shape)
        embedding = tf.reshape(embedding_batched, (10,) + output_shape)

        print("input shape : %s"%(input_shape,))
        print("output shape: %s"%(output_shape,))

        tf.train.export_meta_graph(filename=name + '.meta')

        config = {
            'raw': raw.name,
            'embedding': embedding.name,
            'input_shape': input_shape,
            'output_shape': output_shape}
        with open(name + '.json', 'w') as f:
            json.dump(config, f)

def create_affs(input_shape, intermediate_shape, expected_output_shape, name):

    tf.reset_default_graph()

    with tf.variable_scope('acrlsd'):

        raw = tf.placeholder(tf.float32, shape=input_shape)
        raw_batched = tf.reshape(raw, (1, 1) + input_shape)
        raw_in = tf.reshape(raw_batched, input_shape)
        raw_batched = crop_zyx(raw_batched, (1, 1) + intermediate_shape)
        raw_cropped = tf.reshape(raw_batched, intermediate_shape)

        pretrained_lsd = tf.placeholder(tf.float32, shape=(10,) + intermediate_shape)
        pretrained_lsd_batched = tf.reshape(pretrained_lsd, (1, 10) + intermediate_shape)

        concat_input = tf.concat([raw_batched, pretrained_lsd_batched], axis=1)

        unet, _, _ = mala.networks.unet(
                    concat_input,
                    12,
                    6,
                    [[2,2,2],[2,2,2],[3,3,3]])

        affs_batched, _ = mala.networks.conv_pass(
            unet,
            kernel_sizes=[1],
            num_fmaps=3,
            activation='sigmoid',
            name='affs')
        affs = tf.squeeze(affs_batched, axis=0)

        output_shape = tuple(affs.get_shape().as_list()[1:])
        assert expected_output_shape == output_shape, "%s !=%s"%(expected_output_shape, output_shape)

        gt_affs = tf.placeholder(tf.float32, shape=(3,) + output_shape)
        loss_weights_affs = tf.placeholder(tf.float32, shape=(3,) + output_shape)

        loss = tf.losses.mean_squared_error(
            gt_affs,
            affs,
            loss_weights_affs)

        summary = tf.summary.scalar('acrlsd_eucl_loss', loss)

        opt = tf.train.AdamOptimizer(
            learning_rate=0.5e-4,
            beta1=0.95,
            beta2=0.999,
            epsilon=1e-8)
        optimizer = opt.minimize(loss)

        print("input shape : %s"%(intermediate_shape,))
        print("output shape: %s"%(output_shape,))

        tf.train.export_meta_graph(filename=name + '.meta')

        config = {
            'raw': raw.name,
            'raw_in': raw_in.name,
            'pretrained_lsd': pretrained_lsd.name,
            'affs': affs.name,
            'gt_affs': gt_affs.name,
            'loss_weights_affs': loss_weights_affs.name,
            'loss': loss.name,
            'optimizer': optimizer.name,
            'input_shape': intermediate_shape,
            'output_shape': output_shape,
            'summary': summary.name
            }
        with open(name + '.json', 'w') as f:
            json.dump(config, f)

def create_config(input_shape, output_shape, name):

    config = {
        'input_shape': input_shape,
        'output_shape': output_shape,
        'lsds_setup': 'lsd',
        'lsds_iteration': 300000
        }

    config['outputs'] = {'affs': {"out_dims": 3, "out_dtype": "uint8"}}

    with open(name + '.json', 'w') as f:
        json.dump(config, f)

if __name__ == "__main__":

    train_input_shape = (304, 304, 304)
    train_intermediate_shape = (196, 196, 196)
    train_output_shape = (92, 92, 92)

    create_auto(train_input_shape, train_intermediate_shape, 'train_auto_net')
    create_affs(train_input_shape, train_intermediate_shape, train_output_shape, 'train_net')

    test_input_shape = (364, 364, 364)
    test_output_shape = (260, 260, 260)

    create_affs(test_input_shape, test_input_shape, test_output_shape, 'test_net')

    create_config(test_input_shape, test_output_shape, 'config')

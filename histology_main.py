import tensorflow as tf
import data_processing as data
import model
import output
from constants import BASE_PATH, DATA_PATH, CHKPTS_PATH, GRAPHS_PATH, SEED, NUM_THREADS

if __name__ == '__main__':
    
    # Dimensions of data
    dim1 = 128 
    dim2 = 128 
    dim3 = 3 
    batch_size = 16
    #noise input vector
    z_dim = 100 
    
    # Number of epochs to run
    num_epochs = 1000 
    learning_rate = 0.00025
    beta1 = 0.45
    save_model_every = 100
    save_example_every = 10
    print_every = 100

    with tf.Graph().as_default() as graph1:
        # Random seed for reproducibility
        tf.set_random_seed(SEED)
        
        # Placeholders
        z_batch_tensor = tf.random.uniform((batch_size, z_dim),-1, 1, dtype=tf.float32)
        is_training = tf.placeholder(tf.bool, shape=1)

        # Dataset
        dataset = data.create_dataset(BASE_PATH + DATA_PATH, batch_size, dim1, dim2, dim3, num_epochs)
        iterator = dataset.make_initializable_iterator()
        next_batch = iterator.get_next()
        saveable_obj = tf.data.experimental.make_saveable_from_iterator(iterator)

        # Model
        input_real, input_z, _ = model.model_inputs(dim1, dim2, dim3, z_dim)
        is_training = tf.placeholder(dtype=tf.bool, name='is_training')
        d_loss, g_loss = model.model_loss(input_real, input_z, dim3, is_training)
        d_train_opt, g_train_opt = model.model_opt(d_loss, g_loss, learning_rate, beta1)

        steps = 0
        saver = tf.train.Saver()
        with tf.Session() as sess:
            try:
                saver.restore(sess, BASE_PATH + CHKPTS_PATH + "model.ckpt")
            except:
                sess.run(tf.global_variables_initializer())
            sess.run(iterator.initializer)
            while True:
                steps += 1
                try:
                    # Run optimizers
                    batch = sess.run(next_batch)
                    batch_z = sess.run(z_batch_tensor)
                    sess.run(d_train_opt, feed_dict={input_real: batch, input_z: batch_z,  is_training : True})
                    sess.run(g_train_opt, feed_dict={input_z: batch_z, is_training : True})
                    writer = tf.summary.FileWriter(BASE_PATH + GRAPHS_PATH, sess.graph)
                    if steps % save_model_every == 0:
                        saver.save(sess, BASE_PATH + CHKPTS_PATH + "model.ckpt")
                    if steps % print_every == 0:
                        train_loss_d = d_loss.eval({input_real: batch, input_z: batch_z,is_training : True}, session = sess)
                        train_loss_g = g_loss.eval({input_z: batch_z,is_training : True}, session = sess)
                        print("Epoch Step {}...".format(steps),"Discriminator Loss: {:.4f}...".format(train_loss_d),"Generator Loss: {:.4f}".format(train_loss_g))
                    if steps % save_example_every == 0:
                        output.save_output(sess,z_batch_tensor,input_z,is_training,steps,dim1,dim2,dim3)
                        
                except tf.errors.OutOfRangeError:
                    print("End")
                    break
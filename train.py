import os

import numpy as np
import tensorflow as tf

import aae2
import matplotlib.pyplot as plt
from matplotlib import gridspec, colors
from datetime import datetime
import scipy.io as sio
#from skimage.measure import compare_ssim, compare_psnr
import math
global i
#psnr
def psnr(data_input, reconstruct):
    data_input = (data_input-data_input.min())/(data_input.max()-data_input.min())
    reconstract = (reconstruct-reconstruct.min()) / \
        (reconstruct.max()-reconstruct.min())
    target_data = np.array(data_input)
    ref_data = np.array(reconstruct)
    diff = ref_data - target_data
    diff = diff.flatten('C')
    rmse = math.sqrt(np.mean(diff ** 2.))
    return 20*np.log10(1.0/rmse)

def main1():
                 
    """ prepare data """
    load_fn = './coarse/hydice_urban_pca/train_data.mat'
    load_data = sio.loadmat(load_fn)
    x_input = load_data['train_data']
    x_input = np.array(x_input)  
    x_input = 2*((x_input-x_input.min()) /
                   (x_input.max()-x_input.min()))
    
    #print(x_input.shape)
    load_fn = './data/hydice_urban_pca1.mat'
    load_d = sio.loadmat(load_fn)
    d_input = load_d['d']
    d_input = np.array(d_input)  
    d_input = 2*((d_input-d_input.min()) /
                   (d_input.max()-d_input.min()))
    
    
    load_fn_val = './coarse/hydice_urban_pca/val_data.mat'
    load_data_val = sio.loadmat(load_fn_val)
    x_input_val = load_data_val['val_data']
    x_input_val = np.array(x_input_val)  
    x_input_val = 2*((x_input_val-x_input_val.min()) /
                   (x_input_val.max()-x_input_val.min()))
    
    """ parameters """
   
    tf.reset_default_graph()
    dim_z = 50
    dim_data = x_input.shape[1] 
    #print(dim_data)
    
    batch_size =64
    learn_rate = 1e-3
    #print(batch_size)
    
    learn_rate = 1e-3

    no_epohs =range(5,40,5)
    
    for i in no_epohs:
        n_epochs=i
        """ build graph """
        x = tf.placeholder(tf.float32, shape=[None, dim_data], name='input_data')
        d = tf.placeholder(tf.float32, shape=[dim_data, 1], name='d')
        # dropout
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        # samples drawn from prior distribution
        z_sample = tf.placeholder(
            tf.float32, shape=[None, dim_z], name='prior_sample')
        # network architecture
    
        y, z, R_loss, D_loss, G_loss = aae2.adversarial_autoencoder(
             x, z_sample,d, dim_data, dim_z, keep_prob)
    
        # optimization
        t_vars = tf.trainable_variables()
        di_vars = [var for var in t_vars if "discriminator" in var.name]
        de_vars = [var for var in t_vars if "MLP_decoder" in var.name]
        g_vars = [var for var in t_vars if "MLP_encoder" in var.name]
        ae_vars = g_vars + de_vars
    
        train_op_ae = tf.train.AdamOptimizer(
            learn_rate).minimize(R_loss, var_list=ae_vars)# ae
        train_op_d = tf.train.AdamOptimizer(
            learn_rate/10).minimize(D_loss, var_list=di_vars)#
        train_op_g = tf.train.AdamOptimizer(
            learn_rate).minimize(G_loss, var_list=g_vars)#
    
        """ training """
        saver = tf.train.Saver()
        # train
        with tf.Session() as sess:
    
            writer = tf.summary.FileWriter("logs/", sess.graph)
            sess.run(tf.global_variables_initializer(), feed_dict={keep_prob:1})
            rand_x = np.random.RandomState(42)
            past = datetime.now()  
            for epoch in range(n_epochs):
                rand_x.shuffle(x_input)
                print(x_input.shape[0])
                for batch in np.arange(int(len(x_input) / batch_size)):
                    start = int(batch * batch_size)
                    end = int(start + batch_size)
                    batch_xs_input = x_input[start:end]
                    samples = np.random.randn(dim_data, dim_z)
                    z_id_one_hot_vector = np.ones((dim_z, 1))
    
                #     # reconstruction loss
                    _, r_loss = sess.run(
                        (train_op_ae, R_loss),
                        feed_dict={x: batch_xs_input,d:d_input,z_sample: samples, keep_prob:1})
    
                    # discriminator loss
                    _, d_loss = sess.run(
                        (train_op_d, D_loss),
                        feed_dict={x: batch_xs_input, z_sample: samples, keep_prob: 1})
    
                    # generator loss
                    for _ in range(1):
                        _, g_loss = sess.run(
                            (train_op_g, G_loss),
                            feed_dict={x: batch_xs_input, z_sample: samples, keep_prob: 1})
    
                tot_loss = r_loss + d_loss + g_loss  
    
                # print cost every epoch
                now = datetime.now()
                print("\nEpoch {}/{} - {:.1f}s".format(epoch,
                                                       n_epochs, (now - past).total_seconds()))
                print("Autoencoder Loss: {}".format(np.mean(r_loss)))
    
                print("Discriminator Loss: {}".format(
                    np.mean(d_loss)))
                print("Generator Loss: {}".format(np.mean(g_loss)))
                past = now
                
                
                print("\nValide...")
                reconstruct_train = sess.run(
                    y, feed_dict={x: x_input, keep_prob: 1})
                reconstruct_psnr_train = psnr(
                        x_input, reconstruct_train)
                reproduce_val = sess.run(
                    y, feed_dict={x: x_input_val, keep_prob: 1})
                reconstact_psnr_val = psnr(
                        x_input_val, reproduce_val)
    
                print("Train psnr: {}".format(reconstruct_psnr_train))
                print("Validate psnr: {}".format(reconstact_psnr_val))
                
                
                with open('./log.txt', 'a') as log:
                    log.write("Epoch: {}, iteration: {}\t".format(epoch, 0))
                    log.write("Autoencoder Loss: {}\t".format(
                        np.mean(r_loss)))
                    log.write("Discriminator Loss: {}\t".format(
                        np.mean(d_loss)))
                    log.write("Generator Loss: {}\t".format(np.mean(g_loss)))
                    log.write("train:{}\t".format(
                        reconstruct_psnr_train))
                    log.write("val:{}\n".format(
                        reconstact_psnr_val))
                if epoch % 10 == 0:
                    print("\nSaving models...")
                    saver.save(sess, './model/segundo_pca/model')
                    writer.close()        
def main2():                                      
    sess=tf.Session()
    saver = tf.train.import_meta_graph('./model/hydice_urban_pca/model.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./model/hydice_urban_pca/'))
    
    
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("input_data:0")
    keep_prob = graph.get_tensor_by_name("keep_prob:0")
    y = graph.get_tensor_by_name("MLP_decoder/decode_output:0")
    
    ##########
    load_fn = './data/hydice_urban_pca1.mat'
    load_data = sio.loadmat(load_fn)
    test_data=load_data['data']
    # test_data = load_data['X'].transpose()
    # test_data=test_data.reshape(200,200,189)
    
    #####
    
    
    test_data = np.array(test_data)
    img_band = test_data.shape[2]
    test_data_shape = test_data.shape
    test_data = test_data.reshape([test_data_shape[0]*test_data_shape[1], img_band])
    test_data = 2*((test_data-test_data.min()) /
                        (test_data.max()-test_data.min()))-1
    
    
    reconstruct_result = sess.run(y, feed_dict={x: test_data, keep_prob: 1})
    
    no_epohs =range(5,40,5)
    
    for i in no_epohs:
        
        save_fn = './result/hydice_urban/reconstruct_result_'+str(i)+'.mat'
        sio.savemat(save_fn, {'reconstruct_result': reconstruct_result})
    
if __name__ == '__main__':
    main1()
    main2()

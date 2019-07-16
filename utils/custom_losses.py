import tensorflow as tf
import keras.backend as K

from utils.tensorflow_losses import triplet_semihard_loss
    

def triplet_loss_mult(y_true, y_preds, margin=0.5, n_poses = 4, n_imgs = 40):
    """Triplet semi-hard loss averaged over poses"""
    #Reshape batch (num_img * n_poses, emb_size) to (n_poses, num_img, emb_size)
    y_preds_comb = tf.reshape(y_preds, [n_poses, n_imgs,  -1])
    
    labels = K.slice(y_true, (0, 0), [n_imgs,-1])
       
    temp_loss = lambda x: triplet_semihard_loss(labels, x, margin)
    triplet_losses = tf.map_fn(temp_loss, y_preds_comb)    
    return tf.reduce_mean(triplet_losses, name='triplet_loss_mult')                


def pose_variance(y_true, y_preds, n_poses = 4, n_imgs = 40):
    #Predictions has shape (batch_size, n_poses, emb_size)
    #Compute variance for each example over different poses
    y_preds_comb = tf.reshape(y_preds, [n_poses, n_imgs,  -1])
    _, var = tf.nn.moments(y_preds_comb, axes = [0])
    trace = tf.reduce_sum(var, axis=1)
    loss = tf.reduce_mean(trace, name='pose_variance')
    return loss


def triplet_pose_loss(y_true, y_preds, margin=0.5, weights = [1., 0.01], n_poses = 4, n_imgs = 40):
    """Combined triplet loss (semi-hard for each pose separately) and rotation invariant regularizer"""
    
    triplet_loss =  triplet_loss_mult(y_true, y_preds, margin, n_poses, n_imgs) 
    pose_inv_loss = pose_variance(y_true, y_preds, n_poses, n_imgs)
    
    total_loss = tf.add(weights[0] * triplet_loss, weights[1] * pose_inv_loss, 'triplet_pose_loss')
    return total_loss


def semihard_pose_loss(y_true, y_preds, margin=0.5, weights = [1., 0.01], n_poses = 4, n_imgs = 40):
    """Combined triplet loss (semi-hard over the whole batch) and rotation invariant regularizer"""
    
    triplet_loss =  triplet_semihard_loss(y_true, y_preds, margin) 
    pose_inv_loss = pose_variance(y_true, y_preds, n_poses, n_imgs)
    
    total_loss = tf.add(weights[0] * triplet_loss, weights[1] * pose_inv_loss, 'triplet_pose_loss')
    return total_loss


    
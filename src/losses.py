import tensorflow as tf


# PHM08 asymmetric loss experiment
#
# Used earlier to test PHM08 as a training objective.
def phm08_loss(y_true, y_pred):

    d = y_pred - y_true

    loss = tf.where(
        d >= 0,
        tf.exp(d / 10.0) - 1.0,
        tf.exp(-d / 13.0) - 1.0
    )

    return tf.reduce_mean(loss)


# PHM08 asymmetric score for evaluation
#
# d = predicted RUL - actual RUL
def phm08_score(y_true, y_pred):

    d = y_pred - y_true

    score = tf.where(
        d < 0,
        tf.exp(-d / 13.0) - 1.0,
        tf.exp(d / 10.0) - 1.0
    )

    return tf.reduce_sum(score)
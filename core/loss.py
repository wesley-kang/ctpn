import tensorflow as tf
class Loss(object):
    def build(self,fc_rpn_cls,fc_rpn_pred,label,bbox_targets,bbox_inside_weights,bbox_outside_weights):

        self.rpn_cls = fc_rpn_cls
        self.rpn_pred = fc_rpn_pred
        self.label = label
        self.bbox_targets = bbox_targets
        self.bbox_inside_weights = bbox_inside_weights
        self.bbox_outside_weights = bbox_outside_weights

        rpn_cls_score = tf.reshape(self.rpn_cls, [-1, 2])  # shape (HxWxA, 2)
        rpn_label = self.label
        rpn_label = tf.reshape(rpn_label, [-1])  # shape (HxWxA)

        fg_keep = tf.equal(rpn_label, 1)
        #去掉label为-1的anchor
        rpn_keep = tf.where(tf.not_equal(rpn_label, -1))
        rpn_cls_score = tf.gather(rpn_cls_score, rpn_keep)
        rpn_label = tf.gather(rpn_label, rpn_keep)
        rpn_cross_entropy_n = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=rpn_label, logits=rpn_cls_score)


        rpn_bbox_pred = tf.gather(tf.reshape(self.rpn_pred,[-1,4]), rpn_keep)
        rpn_bbox_targets = tf.gather( tf.reshape(self.bbox_targets,[-1,4]), rpn_keep)
        rpn_bbox_inside_weights = tf.gather(tf.reshape(self.bbox_inside_weights,[-1,4]), rpn_keep)
        rpn_bbox_outside_weights = tf.gather(tf.reshape(self.bbox_outside_weights,[-1,4]), rpn_keep)

        rpn_loss_box_n = tf.reduce_sum(rpn_bbox_outside_weights * self._smooth_l1_dist(
            rpn_bbox_inside_weights * (rpn_bbox_pred - rpn_bbox_targets)), reduction_indices=[1])
        rpn_loss_box = tf.reduce_sum(rpn_loss_box_n) / (tf.reduce_sum(tf.cast(fg_keep, tf.float32)) + 1)
        rpn_cross_entropy = tf.reduce_mean(rpn_cross_entropy_n)
        model_loss = rpn_cross_entropy + rpn_loss_box

        return model_loss,rpn_cross_entropy,rpn_loss_box

    def _smooth_l1_dist(self, deltas, sigma2=9.0):
        deltas_abs = tf.abs(deltas)
        smoothL1_sign = tf.cast(tf.less(deltas_abs, 1.0 / sigma2), tf.float32)
        return tf.square(deltas) * 0.5 * sigma2 * smoothL1_sign +\
               (deltas_abs - 0.5 / sigma2) * tf.abs(smoothL1_sign - 1)


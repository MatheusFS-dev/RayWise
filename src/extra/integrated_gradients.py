import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def plot_integrated_gradients(model, inputs, m_steps=50):
            """
            Compute and plot Integrated Gradients attributions for a two‑input Keras model.
            The target class is assumed to be index 0.
            """
            # Unpack and cast to float32
            lidar_input, coord_input = [tf.cast(x, tf.float32) for x in inputs]
            # Zero baselines
            lidar_base = tf.zeros_like(lidar_input)
            coord_base = tf.zeros_like(coord_input)

            acc_grads_li = tf.zeros_like(lidar_input)
            acc_grads_ci = tf.zeros_like(coord_input)

            for alpha in np.linspace(1.0/m_steps, 1.0, m_steps, dtype=np.float32):
                li = lidar_base + alpha * (lidar_input - lidar_base)
                ci = coord_base + alpha * (coord_input - coord_base)
                with tf.GradientTape() as tape:
                    tape.watch([li, ci])
                    preds = model([li, ci], training=False)[0]
                    score = preds[0]   # first class
                grads_li, grads_ci = tape.gradient(score, [li, ci])
                # replace any None with zeros
                grads_li = grads_li if grads_li is not None else tf.zeros_like(li)
                grads_ci = grads_ci if grads_ci is not None else tf.zeros_like(ci)
                acc_grads_li += grads_li
                acc_grads_ci += grads_ci

            avg_grads_li = acc_grads_li / float(m_steps)
            avg_grads_ci = acc_grads_ci / float(m_steps)

            # attributions
            attr_li = (lidar_input - lidar_base) * avg_grads_li
            attr_ci = (coord_input - coord_base) * avg_grads_ci

            # Plot LiDAR heatmap
            heatmap = tf.reduce_sum(attr_li, axis=-1)[0].numpy()
            plt.figure()
            plt.title("IG Attribution: LiDAR")
            plt.imshow(heatmap, origin="lower")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.show()

            # Plot coordinate bar
            vals = attr_ci[0].numpy()
            plt.figure()
            plt.title("IG Attribution: Coordinates")
            plt.bar(range(len(vals)), vals)
            plt.xlabel("Index")
            plt.ylabel("Attribution")
            plt.show()
            
# # example
# lidar_sample = x_lidar_val[:1]  # Take one sample from validation set
# coord_sample = x_coord_val[:1]  # Take corresponding coordinate sample
# plot_integrated_gradients(model, [lidar_sample, coord_sample], m_steps=100)
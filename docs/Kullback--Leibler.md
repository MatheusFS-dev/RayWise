Temperature-scaled Kullback--Leibler divergence is the part of knowledge distillation that makes a student model imitate the full probability distribution produced by a teacher, not only the single correct class. The idea is simple. A normal classifier outputs logits, which are the raw scores before the softmax. When a temperature $T$ greater than 1 is applied, the softmax becomes softer, so classes that were almost zero now receive visible probability mass. This reveals which wrong classes the teacher still considers plausible, and that extra structure is often called dark knowledge.

Let the teacher logits be $z^{(t)}$ and the student logits be $z^{(s)}$. With temperature $T$, the softened distributions are

$$
p_i^{(t)}(T)=\frac{\exp\left(z_i^{(t)}/T\right)}{\sum_j \exp\left(z_j^{(t)}/T\right)},
\qquad
p_i^{(s)}(T)=\frac{\exp\left(z_i^{(s)}/T\right)}{\sum_j \exp\left(z_j^{(s)}/T\right)}.
$$

The Kullback--Leibler divergence then measures how far the student distribution is from the teacher distribution:

$$
D_{\mathrm{KL}}\left(p^{(t)}(T)\,\|\,p^{(s)}(T)\right)
=
\sum_i p_i^{(t)}(T)\log\frac{p_i^{(t)}(T)}{p_i^{(s)}(T)}.
$$

If the two distributions are identical, this term is zero. If they differ, the value grows. During training, minimizing this loss pushes the student to produce a probability pattern similar to the teacher's.

The role of temperature is important. When $T=1$, the softmax is the usual one, and the distribution may be too sharp. For example, the teacher may place almost all probability on one class, so the student learns little beyond the top class. When $T>1$, the probabilities become smoother. The most likely class still remains important, but the relative similarity among other classes becomes visible. In beam selection, this can be useful because nearby beam pairs may be physically similar, and the teacher may assign non-negligible probability to several strong candidates rather than only one.

A simple example makes this clearer. Suppose there are three classes, and the teacher logits are

$$
[6,\,4,\,1].
$$

With the normal softmax, roughly at $T=1$, the teacher probabilities are approximately

$$
[0.876,\,0.118,\,0.006].
$$

This is very peaked. It mostly says, "class 1 is correct." Now apply temperature $T=2$. The softened teacher distribution becomes approximately

$$
[0.690,\,0.254,\,0.056].
$$

Now the second class is clearly shown as somewhat plausible, and the third class is not completely negligible. This extra structure is what the student tries to imitate.

Now suppose the student logits are

$$
[5,\,3.8,\,1.2].
$$

At $T=2$, the student softmax becomes approximately

$$
[0.595,\,0.326,\,0.079].
$$

The student is reasonably close to the teacher, but not identical. The KL divergence quantifies this mismatch. Computing

$$
D_{\mathrm{KL}}\left(p^{(t)}\,\|\,p^{(s)}\right)
=
0.690\log\frac{0.690}{0.595}
+
0.254\log\frac{0.254}{0.326}
+
0.056\log\frac{0.056}{0.079},
$$

gives a small positive value, which means the student is close but still different. Training reduces this value by adjusting the student logits.

In practice, distillation usually combines two losses. One is the ordinary hard-label cross-entropy with the ground-truth class. The other is the temperature-scaled KL divergence between teacher and student. A common form is

$$
\mathcal{L}
=
(1-\alpha)\,\mathcal{L}_{\mathrm{CE}}
+
\alpha\,T^2\,D_{\mathrm{KL}}\left(p^{(t)}(T)\,\|\,p^{(s)}(T)\right).
$$

Here, $\alpha$ controls how much the student follows the teacher, and $T$ controls how soft the teacher distribution is. The factor $T^2$ is commonly included to keep gradient magnitudes well scaled during optimization.

Conceptually, the hard-label loss teaches the student the correct answer, while the KL term teaches how the teacher ranks all answers. In your case, this means the student does not only learn the optimal beam pair, but also learns which other beam pairs the teacher considers close alternatives. That is why distillation can improve top-$k$ performance even when the student remains compact.

For paper writing, a concise explanation could be:

```tex
The distillation objective combines the conventional hard-label cross-entropy with a temperature-scaled Kullback--Leibler divergence, which matches the softened class-probability distribution of the student to that of the teacher. By using a temperature \(T>1\), the teacher distribution becomes less peaked and reveals relative similarities among non-target classes, allowing the student to learn richer inter-class structure than from one-hot labels alone.
```

If you want, I can also turn this into a shorter IEEE-style paragraph with equations ready for your manuscript.

# Class-Level Anchor Alignment for AlignedNorm

## 1. Design boundary

This version keeps the AlignedNorm training and inference pipeline unchanged and adds one shared anchor bank during training. The new mechanism only compares how the vision and text branches select anchors for the same class.

The original AlignedNorm losses remain active:

- The class-token branch computes cross-entropy from the visual [CLS] feature and the textual EOT feature.
- The learned-token branch computes cross-entropy from the image-side mean representation token and the textual EOT feature.
- The frozen-CLIP directional regularization remains unchanged.
- Representation-token norm alignment remains unchanged. Vision representation tokens align their norms to [CLS], while text representation tokens align their norms to EOT.

The anchor distributions do not modify features, logits, classifiers, or inference. Setting both new loss weights to zero restores the original AlignedNorm objective.

## 2. Class-level inputs

Both encoders summarize their representation tokens at the final layer and project those summaries into the shared CLIP embedding space.

For an image x, the vision branch produces a mean representation vector v_x. For a class c, all vision vectors belonging to that class in the current batch are averaged:

    v_c = mean({v_x | label(x) = c})

The text branch produces one mean representation vector t_c for each class. Thus both anchor queries represent the same class, while the image-specific representation used by the original learned-token classifier remains separate.

If a class appears only once in the batch, its class-level vision vector is necessarily equal to that single image vector.

## 3. Shared anchors and selection

The model learns one anchor bank:

    A = [a_1, ..., a_K]

The anchors live in the shared CLIP space. Vision and text do not receive separate banks or separate selection parameters. Their class-level summaries independently match against the same anchors:

    pi_v(c) = softmax(normalize(v_c) @ normalize(A).T / tau)
    pi_t(c) = softmax(normalize(t_c) @ normalize(A).T / tau)

Each distribution has K elements and describes the selection proportions over the anchor bank. The initial temperature is fixed at tau = 1; it is not learned in this version.

## 4. Added losses

### Class-level selection matching

For every class represented in the batch, the two selection distributions are compared with symmetric KL divergence:

    L_match(c) = 0.5 * (KL(pi_t(c) || pi_v(c)) + KL(pi_v(c) || pi_t(c)))
    L_match = mean_c L_match(c)

This is a class-level constraint. Individual image distributions are not calculated or penalized in this version. The loss updates the representation tokens and anchors so that both modalities select similar anchors for a class, but it does not directly change the classification score.

### Anchor orthogonality

To discourage redundant anchor directions, the normalized anchors are regularized by:

    L_orth = ||normalize(A).T @ normalize(A) - I||_F^2

The anchors are normalized before this calculation, so the penalty controls directional overlap rather than anchor magnitude. It therefore does not compete directly with the existing representation-token norm alignment. Orthogonality does not guarantee that every anchor is used, so training should also record the average selection probability of each anchor.

## 5. Complete objective

The training objective is:

    L = L_AlignedNorm + lambda_match * L_match + lambda_orth * L_orth

L_AlignedNorm contains the original class-token cross-entropy, learned-token cross-entropy, directional regularization, and norm alignment. In particular, the learned-token classification step still compares the image-side mean representation token with the textual EOT feature.

The two new weights should start small so that anchor matching cannot dominate classification. If learned-token accuracy drops, reduce these weights before changing the original AlignedNorm losses.

## 6. Inference and novel classes

No anchor distribution is computed at inference. Base and novel classes both use the original AlignedNorm forward path. Consequently, this version is a training regularizer, not an explicit novel-class anchor router like CASPA.

## 7. Implementation checklist

1. Extract the final-layer vision and text representation-token means and project them into CLIP space.
2. Group vision means by class within each batch and compute v_c.
3. Compute pi_v(c) and pi_t(c) against one shared, learnable anchor bank.
4. Add symmetric KL matching and normalized-anchor orthogonality to the training loss.
5. Leave all original logits, losses, parameters, and evaluation code active and unchanged.
6. Log the mean anchor-selection distribution to detect anchors that remain unused.

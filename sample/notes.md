# Transformer Attention

The attention mechanism lets a model weigh the relevance of every token
against every other token in the same sequence.

## Scaled dot-product attention

Attention is computed as softmax(QK^T / sqrt(d_k)) V. The scaling factor
prevents the dot products from growing large in magnitude, which would push
softmax into regions with vanishing gradients.

## Why it matters

Without attention, sequence models would have to compress all context into a
single fixed vector. Attention lets each position attend directly to any
other position, giving constant-length paths between any two tokens.

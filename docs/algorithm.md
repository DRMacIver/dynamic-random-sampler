# Dynamic Random Sampler: Algorithm Documentation

This document provides a detailed explanation of the data structure and algorithms
implemented in this crate, based on the paper "Dynamic Generation of Discrete Random
Variates" by Matias, Vitter, and Ni (1993/2003).

## Problem Statement

Given $N$ elements with weights $w_1, w_2, \ldots, w_N$, we want to:

1. **Sample**: Generate random index $j$ with probability $\frac{w_j}{\sum_i w_i}$
2. **Update**: Change the weight of element $j$ to a new value $w'_j$

The naive approach (store cumulative weights, binary search) achieves $O(\log N)$ for
sampling but $O(N)$ for updates (rebuilding cumulative sums). This implementation
achieves:

- **Sampling**: $O(\log^* N)$ expected time
- **Updates**: $O(\log^* N)$ amortized expected time

where $\log^* N$ is the iterated logarithm - the number of times you must apply $\log_2$
to $N$ before reaching $\leq 1$. For all practical values of $N$ (up to $2^{65536}$),
$\log^* N \leq 5$.

## Core Data Structure

### Weights in Log Space

All weights are stored internally as $\log_2(w)$ rather than $w$ directly. This provides:

1. **Numerical stability**: Weights spanning from $10^{-300}$ to $10^{300}$ work correctly
2. **Efficient range computation**: Range membership is a simple floor operation
3. **Overflow prevention**: No risk of overflow when computing weight sums

### Ranges

Elements are partitioned into **ranges** $R_j$ based on their weights. Range $R_j$
contains all elements with weights in the interval $[2^{j-1}, 2^j)$.

In log space, this becomes: element with $\log_2(w)$ belongs to range
$j = \lfloor \log_2(w) \rfloor + 1$, i.e., $\log_2(w) \in [j-1, j)$.

**Example**:
- Weight 1.0 ($\log_2 = 0$) goes to range $j=1$
- Weight 2.0 ($\log_2 = 1$) goes to range $j=2$
- Weight 0.5 ($\log_2 = -1$) goes to range $j=0$

### Degrees and Tree Structure

Each range has a **degree** = number of elements it contains.

- **Root ranges**: degree = 1 (single element)
- **Non-root ranges**: degree >= 2 (multiple elements)

Non-root ranges at level $\ell$ become "elements" at level $\ell+1$, where their
"weight" is the total weight of all children. This creates a tree structure:

```
Level 3:  [Root ranges with single children from level 2]
          |
Level 2:  [Ranges containing ranges from level 1]
          |
Level 1:  [Ranges containing actual elements]
          |
Level 0:  [Actual elements with their weights]
```

The tree height is bounded by $O(\log^* N)$ because:
- At each level, only non-root ranges (degree >= 2) propagate up
- Each propagation roughly halves the number of entities

### Level Tables

At each level $\ell$, we maintain a **level table** $T_\ell$ containing all root
ranges at that level. The total weight of $T_\ell$ is the sum of weights of all
its root ranges.

## Sampling Algorithm

The sampling algorithm has three steps:

### Step 1: Select a Level

Select level $\ell$ with probability proportional to the total weight of its
level table $T_\ell$:

$$P(\text{level } \ell) = \frac{\text{weight}(T_\ell)}{\sum_i \text{weight}(T_i)}$$

We use the **Gumbel-max trick** for this selection:
1. For each level $\ell$, compute $\log(\text{weight}(T_\ell)) + G_\ell$ where $G_\ell \sim \text{Gumbel}(0,1)$
2. Return the level with maximum perturbed value

The Gumbel-max trick works in log space, avoiding overflow issues.

### Step 2: Select a Root Range

From the chosen level table $T_\ell$, select a root range $R_j$ proportional to
its total weight. Again we use the Gumbel-max trick for this selection.

### Step 3: Walk Down the Tree

Starting from the selected root range at level $\ell$, walk down to level 1:

For each range $R_j$ at the current level:
1. Use **rejection sampling** to select a child:
   - Pick a random child uniformly
   - Accept with probability $\frac{w_{\text{child}}}{2^j}$
   - If rejected, repeat
2. Move to the selected child at the next lower level
3. Continue until reaching level 1

At level 1, the selected child is an actual element index.

### Rejection Sampling Analysis

Within range $R_j$, all children have weights in $[2^{j-1}, 2^j)$, so:

$$\text{accept probability} = \frac{w_{\text{child}}}{2^j} \geq \frac{2^{j-1}}{2^j} = \frac{1}{2}$$

This guarantees expected 2 trials per rejection step. Combined with the
$O(\log^* N)$ tree height, total sampling time is $O(\log^* N)$.

## Update Algorithm

When element $i$'s weight changes from $w$ to $w'$:

### Basic Update (Section 2.3)

1. If $w'$ is in a different range than $w$:
   - Remove element from old range $R_j$
   - Add element to new range $R_{j'}$
   - Propagate changes up the tree as range weights change

2. If $w'$ is in the same range:
   - Just update the weight within the range
   - Propagate weight changes up

### Section 4 Optimizations

The basic update can have $O(\log N)$ worst-case time. Section 4 optimizations
achieve $O(\log^* N)$ amortized expected time:

#### Tolerance Factor $b$

Instead of moving elements immediately when they cross range boundaries, we
allow a tolerance. Range $R_j$ accepts weights in:

$$[(1-b) \cdot 2^{j-1}, (2+b) \cdot 2^{j-1})$$

An element only changes ranges when its weight moves outside this expanded
interval, requiring a change of at least $b \cdot 2^{j-1}$.

With $b = 0.4$:
- Range $R_j$ normally covers $[2^{j-1}, 2^j)$
- With tolerance, accepts $[0.6 \cdot 2^{j-1}, 2.4 \cdot 2^{j-1})$

#### Degree Bound $d$

Ranges need at least $d$ children to be non-root. This bounds tree structure
changes and ensures amortized efficiency.

Default values: $b = 0.4$, $d = 32$.

## Insert and Delete Operations

### Insert

1. Append new element to the element array
2. Insert into appropriate range at level 1
3. If range transitions from root to non-root, propagate insertion up

### Delete (Soft Delete)

Elements are soft-deleted by setting weight to 0 ($\log_2(w) = -\infty$):

1. Mark element as deleted (weight = NEG_INFINITY)
2. Remove from its range at level 1
3. If range becomes root or empty, propagate deletion up

Soft deletion preserves index stability - existing indices remain valid.

## Implementation Details

### Log-Sum-Exp Trick

To compute $\log(\sum_i w_i)$ from $\log(w_i)$ values without overflow:

$$\log\left(\sum_i w_i\right) = \max_i(\log w_i) + \log\left(\sum_i 2^{\log w_i - \max}\right)$$

### Gumbel-Max Sampling

For categorical sampling from weights $w_i$:

1. Generate $G_i \sim \text{Gumbel}(0,1) = -\log(-\log(U_i))$ where $U_i \sim \text{Uniform}(0,1)$
2. Return $\arg\max_i (\log w_i + G_i)$

This is equivalent to sampling index $i$ with probability $\frac{w_i}{\sum_j w_j}$.

Advantage: Works entirely in log space without normalizing weights.

### Numerical Considerations

- Weights stored as $\log_2(w)$ in `f64`
- Deleted elements use `NEG_INFINITY` as sentinel
- Very small weights (probability < $2^{-1074}$) are effectively zero
- Range numbers are `i32` to support negative ranges (weights < 1)

## Complexity Summary

| Operation | Time Complexity |
|-----------|-----------------|
| Construction | $O(N)$ |
| Sample | $O(\log^* N)$ expected |
| Update (basic) | $O(\log N)$ worst case |
| Update (Section 4) | $O(\log^* N)$ amortized expected |
| Insert | $O(\log^* N)$ amortized expected |
| Delete | $O(\log^* N)$ amortized expected |

Space complexity: $O(N)$

## Optimizations Implemented

### 1. O(1) Random Child Access

The rejection sampling step requires uniform random selection from range children.
Naive `HashMap` iteration with `nth()` is O(n). We use dual storage:

- `children_vec: Vec<Child>` - O(1) random access by index
- `children_idx: HashMap<usize, usize>` - O(1) lookup by child ID

Removal uses swap-remove to maintain O(1) operations:
1. Look up position via `children_idx`
2. Swap-remove from `children_vec`
3. Update moved element's position in `children_idx`

**Impact**: Up to 83% faster sampling for uniform distributions with 1000 elements.

### 2. Gumbel-Max for Level/Range Selection

Instead of explicit normalization and binary search, we use the Gumbel-max trick
for selecting levels and root ranges. This:
- Avoids computing total weights
- Works in log space (no overflow)
- Is numerically stable for extreme weight ranges

### 3. Weight Caching

Range total weights are cached and invalidated on modification, avoiding
redundant log-sum-exp computations.

### 4. Lazy Propagation (Section 4)

With tolerance factor $b$, small weight changes don't propagate through the
tree, reducing amortized update cost.

## Benchmark Results

After optimization (measured on development machine):

| Benchmark | Before | After | Improvement |
|-----------|--------|-------|-------------|
| single_sample/uniform/1000 | 1175ns | 198ns | 83% faster |
| single_sample/power_law/1000 | 501ns | 362ns | 28% faster |
| batch_1000/uniform/1000 | 1165us | 199us | 83% faster |
| construction/uniform/1000 | 169us | 135us | 20% faster |

## References

Matias, Y., Vitter, J. S., & Ni, W. (2003). "Dynamic generation of discrete
random variates." Theory of Computing Systems, 36(4), 329-358.

Original conference version: SODA 1993.

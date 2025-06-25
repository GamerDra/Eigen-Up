# Subspace Recommendation Engine

This project uses concepts from MIT 18.06 Linear Algebra (Lectures 9, 10, 11, 14, 15) to implement a recommendation system simulator based purely on linear algebra. No machine learning libraries, just math.

---

## 📊 Motivation

Recommendation systems typically project users into latent vector spaces and approximate preferences. In this project, we:

* Analyze linear dependence between users,
* Decompose user behavior using subspaces (column space, null space),
* Use orthogonal projections to estimate preferences,
* Use rank-1 approximations to compress behavior data,
* Visualize relationships between users using graph theory.

---

## 🔬 Concepts Covered

| Concept                    | Usage                                 |
| -------------------------- | ------------------------------------- |
| **Linear Independence**    | Detect redundant users                |
| **Column/Row/Null Spaces** | Analyze structure of user-item matrix |
| **Rank-1 Approximation**   | Simulate simplified user behavior     |
| **Orthogonal Projections** | Estimate preferences from a subspace  |
| **Residuals & Errors**     | Evaluate projection effectiveness     |

---

## 📚 Dataset

We use a synthetic user-item rating matrix:

```python
import numpy as np

A = np.array([
    [5, 3, 0, 1, 0],
    [4, 0, 0, 1, 1],
    [1, 1, 0, 5, 0],
    [0, 0, 5, 4, 5],
    [0, 1, 5, 4, 4],
    [1, 0, 0, 2, 1]
])
```

Each row = user, column = item, value = rating (0-5).

---

## 🔧 Setup

```bash
# Create a virtual environment (optional)
python3 -m venv venv
source venv/bin/activate

# Install required libraries
pip install numpy scipy matplotlib seaborn networkx
```

---

## 📚 Tasks

### 1. `check_independence.py`

* Check linear dependence among user rating vectors.
* Determine rank of the user matrix.
* Identify a minimal set of independent users that span the space.

### 2. `project_user.py`

* Select trusted users (e.g., users 0, 1, 2).
* Project a new user's ratings into their subspace using orthogonal projection.
* Compute residual and check orthogonality.
* Plot original vs projected rating vector.

### 3. `rank1_approx.py`

* Compute best rank-1 approximation of A using SVD.
* Reconstruct a simplified matrix `A_1`.
* Compare Frobenius norm of difference `||A - A_1||`.
* Visualize full vs low-rank rating heatmaps.

### 4. `four_spaces.py`

* Compute:

  * Column space basis
  * Row space basis
  * Null space
  * Left null space
* Verify Fundamental Theorem:

  * `Col(A)` orthogonal to `Null(A^T)`
  * `Row(A)` orthogonal to `Null(A)`

### 5. `user_cluster_graph.py`

* Use rank-1 approximation to derive user similarity.
* Construct graph with edge weights as similarity.
* Visualize network using `networkx`.

---

## 📈 Sample Outputs

* Heatmaps of original vs approximated rating matrices.
* Graph plot showing user similarity clusters.
* Plots of original, projected, and residual vectors.
* Matrix ranks and dimensions printed to console.

---

## ✅ Run

```bash
python check_independence.py
python project_user.py
python rank1_approx.py
python four_spaces.py
python user_cluster_graph.py
```

---

## 🤔 Bonus Challenge

Can you find a user whose ratings lie entirely in the null space of another user set's column space? What does that imply about their similarity?

---

## 📚 References

* MIT 18.06 Linear Algebra

  * Lecture 9: Independence, Basis, Dimension
  * Lecture 10: The Four Fundamental Subspaces
  * Lecture 11: Rank-1 matrices, Small-World Graphs
  * Lecture 14: Orthogonal Subspaces
  * Lecture 15: Projections
* Gilbert Strang's textbook: *Introduction to Linear Algebra*

# LDA Project

Final project for the course Stats 601 - Statistical Learning,
taken at the University of Michigan in Spring 2025 term.
Implement and run the Variational Inference fitting algorithm for LDA
as describe in the Latent Dirichlet Allocation paper by Blei et. al.

Available make commands:

* `make`: Run the parameter estimation experiment and the perplexity experiment.
* `make notebooks`: Generate `.ipynb` notebook files to play with.
* `make estimation` and `make perplexity`: run the two experiments separately.
* `make profile-estimation` and `make profile-perplexity`: Run, with the cProfile profiler, the unvectorized and vectorized algorithm for the two experiments
* `make clean`: Clean up outputs, virtual environments, and artifacts.

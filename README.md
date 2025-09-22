# LDA Project

Final project for the course Stats 601 - Statistical Learning,
taken at the University of Michigan in Spring 2025 term.
Implement and run the Variational Inference fitting algorithm for LDA
as describe in the Latent Dirichlet Allocation paper by Blei et. al.

Run the test file with `python3 -m pytest src/experiments/testcorrectness.py`.
Run the experiments with `python3 src/experiments/experiment.py`.

The dependencies are numpy, matplotlib, sklearn, scipy and tqdm. To make sure, 
create a virtual environment with `python3 -m venv venv`. Activate virtual 
environment with `source venv/bin/activate` and run 
`pip3 install -r requirements.txt`. 
Use `deactivate` to deactivate the virtual environment.

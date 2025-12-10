.PHONY: all venv test make-notebooks \
		profile-estimation run-estimation \
		profile-perplexity run-perplexity \
		clean-venv clean-generated clean \
		clean-outputs clean-artifacts clean-notebooks


# Abbreviations
PY := python3
PIP := python3 -m pip 
VENV_DIR := .venv
VENV_PY := $(VENV_DIR)/bin/$(PY)
VENV_PIP := $(VENV_DIR)/bin/$(PIP)
RESULT_DIR := output

# Default target first
all : $(VENV_DIR) run-estimation run-perplexity

# Virtual environment
$(VENV_DIR) : requirements.txt # check requirement changes
	$(PY) -m venv $(VENV_DIR)
	$(VENV_PIP) install -r requirements.txt
	touch $(VENV_DIR)

venv : $(VENV_DIR)

test : $(VENV_DIR)
	$(VENV_PY) -m pytest src/experiments/testcorrectness.py

make-notebooks : $(VENV_DIR)
	$(VENV_PY) -m jupytext --to ipynb src/notebooks/Notebook1ParamEst.py -o Notebook1ParamEst.ipynb
	$(VENV_PY) -m jupytext --to ipynb src/notebooks/Notebook2Perplexity.py -o Notebook2Perplexity.ipynb
	$(VENV_PY) -m jupytext --to ipynb src/notebooks/Notebook3Classification.py -o Notebook3Classification.ipynb

profile-estimation : $(VENV_DIR)
	mkdir -p profile
	mkdir -p $(RESULT_DIR)
	$(VENV_DIR)/bin/pyinstrument -m src.scripts.run_param_est -o output/temp_est1.jpg -u > profile/profile_estimation.txt
	$(VENV_DIR)/bin/pyinstrument -m src.scripts.run_param_est -o output/temp_est2.jpg >> profile/profile_estimation.txt

profile-perplexity : $(VENV_DIR)
	mkdir -p profile
	mkdir -p $(RESULT_DIR)
	$(VENV_DIR)/bin/pyinstrument -m src.scripts.run_perplexity_exp -o output/temp_per1.jpg -u > profile/profile_perplexity.txt
	$(VENV_DIR)/bin/pyinstrument -m src.scripts.run_perplexity_exp -o output/temp_per2.jpg >> profile/profile_perplexity.txt

run-estimation : $(VENV_DIR)
	mkdir -p $(RESULT_DIR)
	$(VENV_PY) -m src.scripts.run_param_est -o output/fitted_parameters.jpg

run-perplexity : $(VENV_DIR)
	mkdir -p $(RESULT_DIR)
	$(VENV_PY) -m src.scripts.run_perplexity_exp -o output/plot_perplexity.jpg


# Clean ups
clean : clean-venv clean-generated

clean-generated : clean-outputs clean-artifacts clean-notebooks

clean-notebooks :
	@rm -f Notebook1ParamEst.ipynb
	@rm -f Notebook2Perplexity.ipynb
	@rm -f Notebook3Classification.ipynb

clean-venv : 
	@rm -rf $(VENV_DIR)

clean-outputs :
	@rm -rf $(RESULT_DIR)
	@rm -rf profile

clean-artifacts : 
	@find . -type f -name "prof.pstats" -delete
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
	@find . -type d -name ".pytest_cache"  -exec rm -rf {} +

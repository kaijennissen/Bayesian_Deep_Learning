.ONESHELL:
.PHONY: install-dev
ENV:=BNN
CONDA_ACTIVATE = source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

create:
	conda create -c conda-forge -n $(ENV) -y python=3.9

install-dev:
	$(CONDA_ACTIVATE) $(ENV)
	pip install -r requirements-dev.txt

install:
	$(CONDA_ACTIVATE) $(ENV)
	pip install -r requirements.txt

update:
	$(CONDA_ACTIVATE) $(ENV)
	pip-compile --resolver=backtracking requirements-dev.in
	pip-compile --resolver=backtracking requirements.in

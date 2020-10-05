python -m isort -y -sp .
python -m py.test --pylint --pylint-rcfile=./pylint.rc --pylint-error-types=EFCW --pylint-jobs=4
python tests.py --verbose

python3 -m isort -y -sp .
python3 -m pytest --pylint --pylint-rcfile=./pylint.rc --pylint-error-types=EFCRW --pylint-jobs=4
python3 tests.py --verbose

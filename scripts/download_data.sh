#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

pushd $DIR/../data/

mkdir -p kaggle; pushd kaggle
wget https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv

mkdir -p titanic; pushd titanic
echo Download the following files from https://www.kaggle.com/c/titanic/data to this directory, after logging in:
echo   Directory: $PWD
echo     train.csv
echo     test.csv
popd

popd

mkdir -p uci; pushd uci
wget https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
wget https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test
popd

mkdir -p odds; pushd odds
wget https://www.openml.org/data/get_csv/52214/phpn1jVwe
mv phpn1jVwe mammography.csv
popd

popd

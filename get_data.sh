#!/usr/bin/env bash

echo "=== Acquiring datasets ==="
echo "---"

if [ -d "$PWD/data/" ]
then
  echo "Directory /data/ exists."
else
  echo "Create Directory /data."
  mkdir data
fi

# (1) SST-2
echo "Downloading SST-2 Dataset..."
if [ -d "$PWD/data/SST-2" ]
then
  echo "Directory /data/SST-2 exists."
else
  echo "Directory /data/SST-2 does not exist."
  python utilities/download_glue.py --tasks SST
fi
echo "---"

# (2) QQP
echo "Downloading QQP Dataset..."
if [ -d "$PWD/data/QQP" ]
then
  echo "Directory /data/QQP exists."
else
  echo "Directory /data/QQP does not exist."
  python utilities/download_glue.py --tasks QQP
fi
echo "---"

# (3) QNLI
echo "Downloading QNLI Dataset..."
if [ -d "$PWD/data/QNLI" ]
then
  echo "Directory /data/QNLI exists."
else
  echo "Directory /data/QNLI does not exist."
  python utilities/download_glue.py --tasks QNLI
fi
echo "---"

# (4) PubMed
# (5) IMDB
# (6) AG_NEWS
# (7) DBPEDIA


#echo "Downloading AG_NEWS..."
## I have to fix that
#if [ -d "$PWD/data/AG_NEWS" ]
#then
#  echo "Directory /data/AG_NEWS exists."
#else
#  echo "Directory /data/AG_NEWS does not exist."
#  cd data
#  mkdir AG_NEWS
#  cd AG_NEWS
#  https://drive.google.com/file/d/1X8hXEEpVscCVPsQnBKZZjozoUP7QL-mI/view?usp=sharing
#  wget https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbUDNpeUdjb0wxRms #-O "ag_news_csv.tar.gz"
#  tar -xzvf "ag_news_csv.tar.gz" -C "${DATADIR}"
#  cd ../../
#fi
#echo "---"

#echo "Downloading DBPedia..."
#if [ -d "$PWD/datasets/DBPEDIA" ]
#then
#  echo "Directory /datasets/DBPEDIA exists."
#else
#  cd datasets
#  mkdir DBPEDIA
#  cd DBPEDIA
#  wget https://drive.google.com/file/d/0Bz8a_Dbh9QhbQ2Vic1kxMmZZQ1k
#fi

echo "---"
echo "The End :)"

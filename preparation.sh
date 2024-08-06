#!/bin/bash


# downloads the datasets
cd data

# IMPPRES
wget https://github.com/facebookresearch/Imppres/raw/main/dataset/IMPPRES.zip
unzip IMPPRES.zip
rm IMPPRES.zip
rm -r __MACOSX  # if you want to remove the __MACOSX folder

# PROPRES


# NOPE

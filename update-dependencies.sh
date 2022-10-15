#!/bin/sh
pipreqs --force --savepath $1
pip3 freeze | grep "pipreqs" >> $1

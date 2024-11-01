#!/bin/bash

if [[ $1 = "-a" ]]; then  # copy all files in src/ dir
  scp -i ~/.ssh/azure -r src/*.py nskapura@172.210.252.176:/home/nskapura/neural/src
elif [[ $# -gt 0 ]]; then # copy list of files
  scp -i ~/.ssh/azure "$@" nskapura@172.210.252.176:/home/nskapura/neural/src
else  # copy only changes in git repo
  scp -i ~/.ssh/azure `git status --porcelain | awk 'match($1, "M"){print $2}'` nskapura@172.210.252.176:/home/nskapura/neural/src
fi

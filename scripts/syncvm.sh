#!/bin/bash

if [[ $1 = "-a" ]]; then
  scp -i ~/.ssh/azure -r src/*.py nskapura@172.210.252.176:/home/nskapura/neural/src
else
  scp -i ~/.ssh/azure `git status --porcelain | awk 'match($1, "M"){print $2}'` nskapura@172.210.252.176:/home/nskapura/neural/src
fi

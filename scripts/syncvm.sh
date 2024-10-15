#!/bin/bash

if [[ $1 = "-a" ]]; then
  scp -i ~/.ssh/azure -r src/*.py nskapura@172.210.252.176:/home/nskapura/neural/src
elif [[ -n $1 ]]; then
  scp -i ~/.ssh/azure $1 nskapura@172.210.252.176:/home/nskapura/neural/$2
else
  scp -i ~/.ssh/azure `git status --porcelain | awk 'match($1, "M"){print $2}'` nskapura@172.210.252.176:/home/nskapura/neural/src
  echo updates
fi

#!/bin/bash

scp `git status --porcelain | awk 'match($1, "M"){print $2}'` nskapura@172.210.252.176:/home/nskapura/neural/src
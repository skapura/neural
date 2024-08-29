#!/bin/bash

az vm start -g rg-main -n vm-tf
az vm stop -g rg-main -n vm-tf
az vm deallocate -g rg-main -n vm-tf
az vm list -d -o table

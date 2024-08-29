from keras import models
from paramiko import SSHClient
from scp import SCPClient
import sys
import os
import const


def put(source, dest='', host=const.AZURE_HOST, user=const.AZURE_USER):
    ssh = SSHClient()
    ssh.load_system_host_keys()
    ssh.connect(host, username=user)
    scp = SCPClient(ssh.get_transport())
    scp.put(source, const.AZURE_ROOT + dest)
    scp.close()


def get(source, dest='.', host=const.AZURE_HOST, user=const.AZURE_USER):
    ssh = SSHClient()
    ssh.load_system_host_keys()
    ssh.connect(host, username=user)
    scp = SCPClient(ssh.get_transport())
    scp.get(const.AZURE_ROOT + source, dest)
    scp.close()


def shell(command, host=const.AZURE_HOST, user=const.AZURE_USER):
    ssh = SSHClient()
    ssh.load_system_host_keys()
    ssh.connect(host, username=user)
    chan = ssh.invoke_shell(term='bash', width=80, height=24, width_pixels=0, height_pixels=0, environment=None)
    stdin = chan.makefile('wb')
    stdin.write(command + '\nexit\n')
    stdin.flush()
    while True:
        bytes = chan.recv(1024)
        if len(bytes) == 0:
            break
        sys.stdout.buffer.write(bytes)
    ssh.close()


def execute(command, host=const.AZURE_HOST, user=const.AZURE_USER):
    ssh = SSHClient()
    ssh.load_system_host_keys()
    ssh.connect(host, username=user)
    stdin, stdout, stderr = ssh.exec_command(command)
    while True:
        bytes = stdout.channel.recv(1024)
        if len(bytes) == 0:
            break
        sys.stdout.buffer.write(bytes)
    ssh.close()


def train(spec_path='src/models/train_spec.py', model_path=None, host=const.AZURE_HOST, user=const.AZURE_USER):
    base = os.path.basename(spec_path)
    put(spec_path, base, host, user)    # Copy script to vm
    if model_path is not None:
        put(model_path, os.path.basename(model_path), host, user)
    shell('python3 ' + base, host, user)    # train model
    get('trained.keras', '.model.keras', host, user)    # copy model to local
    delcmd = 'rm -f trained.keras ' + base
    if model_path is not None:
        delcmd += ' ' + os.path.basename(model_path)
    execute(delcmd)  # Delete files from vm
    model = models.load_model('.model.keras', compile=True)
    os.remove('.model.keras')   # delete local copy of model
    return model

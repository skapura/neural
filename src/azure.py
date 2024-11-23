from keras import models
from paramiko import SSHClient
from scp import SCPClient
import pickle
import inspect
import ast
import astunparse
import sys
import os
import const


class AzureSession:
    def __init__(self):
        self.ssh = None
        self.host = None
        self.user = None
        self.chan = None
        self.stdin = None

    def open(self, host=const.AZURE_HOST, user=const.AZURE_USER, working_dir=None):
        self.ssh = SSHClient()
        self.host = host
        self.user = user
        self.ssh.load_system_host_keys()
        self.ssh.connect(host, username=user)
        self.chan = self.ssh.invoke_shell(term='bash', width=80, height=24, width_pixels=0, height_pixels=0, environment=None)
        self.stdin = self.chan.makefile('wb')
        self.flush()
        if working_dir is not None:
            self.execute('cd ' + working_dir)

    def close(self):
        self.stdin.write('exit\n')
        self.stdin.flush()
        self.ssh.close()
        self.ssh = None
        self.host = None
        self.user = None
        self.chan = None
        self.stdin = None

    def flush(self):
        while True:     # TODO: will hang if prompt divided between 2 buffer reads
            buf = self.chan.recv(1024)
            bufstr = str(buf).split('\\r\\n')
            sys.stdout.buffer.write(buf)
            if len(buf) == 0 or 'nskapura@vm-tf:' in bufstr[-1]:
                break

    def execute(self, command):
        self.stdin.write(command + '\n')
        self.stdin.flush()
        self.flush()

    def put(self, source, dest=''):
        scp = SCPClient(self.ssh.get_transport())
        scp.put(source, const.AZURE_ROOT + dest)
        scp.close()

    def get(self, source, dest='.'):
        scp = SCPClient(self.ssh.get_transport())
        scp.get(const.AZURE_ROOT + source, dest)
        scp.close()

    def upload_function(self, func_def):
        path = 'src/remote_spec.py'
        runcode = replace_func(path, func_def, 'run')
        with open('temp/temp_spec.py', 'w') as file:
            file.write(runcode)
        self.put('temp/temp_spec.py', dest='neural/' + path)


def replace_func(path, func_def, func_name='load_data'):

    # Parse function def
    fcode = inspect.getsource(func_def)
    rt = ast.parse(fcode).body[0]
    rt.name = 'run'

    # Parse code file
    with open(path, 'r') as file:
        scode = file.read()
    spectree = ast.parse(scode)

    # Replace function
    class ReplaceFunctionDef(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            if node.name == func_name:
                new_node = rt
                return new_node
            return node

    transformer = ReplaceFunctionDef()
    newtree = transformer.visit(spectree)
    newspec = astunparse.unparse(newtree)
    return newspec


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

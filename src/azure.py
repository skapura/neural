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

    def train(self, model, epochs=1, data_loader=None):
        self.upload_spec('src/train_spec.py', data_loader)
        model.save('session/temp_model.keras')
        self.put('session/temp_model.keras', dest='neural/session/temp_model.keras')
        self.execute('python src/train_spec.py ' + str(epochs))
        self.get('neural/session/temp_model.keras', dest='session/temp_model.keras')
        trained = models.load_model('session/temp_model.keras', compile=True)
        return trained

    def evaluate(self, model, data_loader=None):
        self.upload_spec('src/eval_spec.py', data_loader)
        model.save('session/temp_model.keras')
        self.put('session/temp_model.keras', dest='neural/session/temp_model.keras')
        self.execute('python src/eval_spec.py')
        self.get('neural/session/results.pkl', dest='session/results.pkl')
        with open('session/results.pkl', 'rb') as file:
            results = pickle.load(file)
        return results

    def upload_spec(self, path, func_def=None):
        if func_def is None:
            self.put(path, dest='neural/' + path)
        else:
            evalcode = replace_func(path, func_def)
            with open('session/temp_spec.py', 'w') as file:
                file.write(evalcode)
            self.put('session/temp_spec.py', dest='neural/' + path)


def replace_func(path, func_def):

    # Parse function def
    fcode = inspect.getsource(func_def)
    rt = ast.parse(fcode).body[0]

    # Parse code file
    with open(path, 'r') as file:
        scode = file.read()
    spectree = ast.parse(scode)

    # Replace function
    class ReplaceFunctionDef(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            if node.name == 'load_data':
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

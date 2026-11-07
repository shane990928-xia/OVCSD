"""Use one lightweight process per unchanged ALFWorld worker, without Ray RPC pools."""
import atexit
import multiprocessing
import os
import threading
import traceback
import cloudpickle


def worker_main(connection,payload):
    os.environ['CUDA_VISIBLE_DEVICES']=''
    worker=None
    try:
        constructor,args,kwargs=cloudpickle.loads(payload)
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        worker=constructor(*args,**kwargs)
        connection.send(('ready',None))
        while True:
            sequence,method,args,kwargs=connection.recv()
            if method=='__close__':
                worker._close_env();connection.send((sequence,True,None));return
            try:connection.send((sequence,True,getattr(worker,method)(*args,**kwargs)))
            except BaseException:connection.send((sequence,False,traceback.format_exc()))
    except EOFError:
        pass
    except BaseException:
        try:connection.send(('initialization_error',traceback.format_exc()))
        except Exception:pass
    finally:
        if worker is not None:worker._close_env()
        connection.close()


class Result:
    def __init__(self,actor,sequence):self.actor,self.sequence=actor,sequence
    def get(self):return self.actor.receive(self.sequence)


class Method:
    def __init__(self,actor,name):self.actor,self.name=actor,name
    def remote(self,*args,**kwargs):return self.actor.send(self.name,args,kwargs)


class Actor:
    def __init__(self,constructor,args,kwargs):
        context=multiprocessing.get_context('spawn')
        parent,child=context.Pipe()
        self.connection=parent;self.sequence=0;self.pending={};self.ready=False;self.closed=False
        self.lock=threading.RLock()
        self.process=context.Process(target=worker_main,args=(child,cloudpickle.dumps((constructor,args,kwargs))),daemon=False)
        self.process.start();child.close()
        atexit.register(self.close)
    def __getattr__(self,name):
        if name.startswith('_'):raise AttributeError(name)
        return Method(self,name)
    def ensure_ready(self):
        if not self.ready:
            if not self.connection.poll(300):raise TimeoutError('ALFWorld process initialization timeout')
            message=self.connection.recv()
            if message[0]!='ready':raise RuntimeError(message)
            self.ready=True
    def send(self,name,args,kwargs):
        with self.lock:
            self.ensure_ready();sequence=self.sequence;self.sequence+=1
            self.connection.send((sequence,name,args,kwargs));return Result(self,sequence)
    def receive(self,sequence):
        with self.lock:
            while sequence not in self.pending:
                if not self.connection.poll(300):raise TimeoutError('ALFWorld process response timeout')
                index,okay,value=self.connection.recv();self.pending[index]=(okay,value)
            okay,value=self.pending.pop(sequence)
            if not okay:raise RuntimeError(value)
            return value
    def close(self):
        if self.closed:return
        self.closed=True
        if self.process.is_alive():
            try:
                self.ensure_ready();sequence=self.sequence;self.connection.send((sequence,'__close__',(),{}))
                if self.connection.poll(5):self.connection.recv()
            except Exception:pass
            self.process.join(timeout=3)
            if self.process.is_alive():self.process.terminate();self.process.join(timeout=3)
        self.connection.close()


class Factory:
    def __init__(self,constructor):self.constructor=constructor
    def remote(self,*args,**kwargs):return Actor(self.constructor,args,kwargs)


class RayEnvironmentProxy:
    """Only this env module sees this proxy; GPU Ray dispatch is untouched."""
    def __init__(self,ray,constructor):self.actual,self.constructor=ray,constructor
    def is_initialized(self):return True
    def init(self,*args,**kwargs):return None
    def remote(self,*args,**kwargs):
        def decorate(constructor):
            return Factory(constructor) if constructor is self.constructor else self.actual.remote(**kwargs)(constructor)
        return decorate(args[0]) if args else decorate
    def get(self,value,*args,**kwargs):
        if isinstance(value,Result):return value.get()
        if isinstance(value,(list,tuple)):
            return [self.get(item,*args,**kwargs) for item in value]
        return self.actual.get(value,*args,**kwargs)
    def kill(self,actor,*args,**kwargs):
        if isinstance(actor,Actor):return actor.close()
        return self.actual.kill(actor,*args,**kwargs)
    def __getattr__(self,name):return getattr(self.actual,name)

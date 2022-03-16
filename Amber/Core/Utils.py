from typing import Union, Callable, List, Tuple
import threading
import multiprocessing


class ParallelException:
    def __init__(self, msg: str):
        self.msg = msg

    def __str__(self):
        return self.msg


def task_async(f, *args, **kwargs):
    threading.Thread(target=f, args=args, kwargs=kwargs).start()


def parallel(funcs: Union[Callable, List[Callable]], paras: Union[Tuple, List[Tuple]] = None):
    if not paras:
        paras = ()
    if isinstance(funcs, List):
        if isinstance(paras, List):
            if len(funcs) != len(paras):
                raise ParallelException("Functions and Parameters number not match: {} vs {}".
                                        format(len(funcs), len(paras)))
            else:
                pass
        else:
            paras = [paras for _ in funcs]
    else:
        if isinstance(paras, List):
            funcs = [funcs for _ in paras]
        else:
            funcs = [funcs]
            paras = [paras]

    outputs = [None for _ in funcs]
    exceptions = [None for _ in funcs]

    def run(i: int, outputs: list, exceptions: list):
        try:
            outputs[i] = funcs[i](*paras[i])
        except Exception as e:
            exceptions[i] = e

    threads = []
    for i in range(len(funcs)):
        threads.append(threading.Thread(target=run, args=(i, outputs, exceptions)))
        threads[-1].start()

    for thread in threads:
        thread.join()

    return outputs, exceptions


def parallel_process(funcs: Union[Callable, List[Callable]], paras: Union[Tuple, List[Tuple]] = None):
    if not paras:
        paras = ()
    if isinstance(funcs, List):
        if isinstance(paras, List):
            if len(funcs) != len(paras):
                raise ParallelException("Functions and Parameters number not match: {} vs {}".
                                        format(len(funcs), len(paras)))
            else:
                pass
        else:
            paras = [paras for _ in funcs]
    else:
        if isinstance(paras, List):
            funcs = [funcs for _ in paras]
        else:
            funcs = [funcs]
            paras = [paras]

    manager = multiprocessing.Manager()
    outputs = manager.list([None for _ in funcs])
    exceptions = manager.list([None for _ in funcs])

    def run(i: int, outputs: list, exceptions: list):
        try:
            outputs[i] = funcs[i](*paras[i])
        except Exception as e:
            exceptions[i] = e

    processes = []
    for i in range(len(funcs)):
        processes.append(multiprocessing.Process(target=run, args=(i, outputs, exceptions)))
        processes[-1].start()

    for process in processes:
        process.join()

    return outputs, exceptions

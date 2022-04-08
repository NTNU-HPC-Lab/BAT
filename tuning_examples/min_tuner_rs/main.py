import json
from itertools import product
from random import randint
import multiprocessing as mp
import os

from tuning_examples.common.benchmark_helpers import compile_benchmark, run_benchmark
from tuning_examples.common.helpers2 import result_builder
from tuning_examples.min_tuner.progressbar import Progressbar


class MinTuner:

    def __init__(self, args):
        with open(args.path, 'r') as f:
            self.oadc = json.load(f)

    @staticmethod
    def cartesian_space_builder(cfg_def):
        params = []
        for param in cfg_def['parameters']:
            v = param['values']
            if v.get('value_list', None):
                v_list = list(v['value_list'])
                params.append(v_list)
            if v.get('value_range', None):
                v_range = list(range(int(v['value_range']['start']),
                                     int(v['value_range']['end']) + int(v['value_range']['inclusive'] == True),
                                     int(v['value_range']['stride'])))
                params.append(v_range)
        return list(product(*params))

    @staticmethod
    def list_to_dict(cfg_def, cfg_list):
        d = []
        for i, param in enumerate(cfg_def['parameters']):
            r = dict()
            r['name'] = param['name']
            r['value'] = cfg_list[i]
            d.append(r)
        return d

    def brute_force(self, cfg_def, i):
        cartesian_space = self.cartesian_space_builder(cfg_def)
        return self.list_to_dict(cfg_def, cartesian_space[i])

    def random_search(self, cfg_def):
        cartesian_space = self.cartesian_space_builder(cfg_def)
        return self.list_to_dict(
            cfg_def, cartesian_space[randint(0, cfg_def['cardinality'])])

    def pick_config(self, cfg_def, search_settings, i):
        name = search_settings['search_technique']['name']
        if name == "brute_force":
            return self.brute_force(cfg_def, i)
        elif name == "random_search":
            return self.random_search(cfg_def)

    def run(self):
        all_results = []
        meta = self.oadc['metadata']
        percent = 0.01
        print("hello")
        r = range(int(float(meta['search_settings']['budget']['steps']) * percent))
        print(r)
        prog = Progressbar(len(r))
        prog.algorithm(meta['benchmark_suite']['benchmark_name'])

        n_gpus = 1
        p_gpus = []
        q_list = []

        for gpu in range(n_gpus):
            q_list.append(mp.Queue())
            p_gpus.append(mp.Process(target=self.inner_loop, args=(meta, all_results, gpu,
                                                                   range(gpu, len(r), n_gpus), prog, q_list[gpu])))
            p_gpus[gpu].start()
        for gpu in range(n_gpus):
            p_gpus[gpu].join()
            all_results.append(q_list[gpu].get())

        self.inner_loop(meta, all_results, 0, range(len(r) - (len(r) % n_gpus), len(r)), prog, None)
        return all_results

    def inner_loop(self, meta, all_results, gpu, r, prog, q):
        print('module name:', __name__)
        print('parent process:', os.getppid())
        print('process id:', os.getpid())
        print("Worker: ", gpu)
        local_results = []
        for i in r:
            cfg = self.pick_config(meta['configuration_space'], meta['search_settings'], i)
            compile_result = compile_benchmark(meta, cfg, gpu)
            run_result = run_benchmark(meta, gpu)
            result = result_builder(compile_result, run_result, cfg)
            local_results.append(result)
            prog.update_progress(i)
        if q is not None:
            q.put(local_results)
        else:
            all_results.extend(local_results)

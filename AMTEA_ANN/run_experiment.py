from ann_lib import *
from cea import cea
from amtea import amtea
from utils.tools import *
from helpers import *
import yaml

# load benchmark
singletask_benchmark = yaml.load(open('ann_lib/singletask-benchmark.yaml'))

# load config
config = get_config('config.yaml')
db = config['database']
conn = create_connection(db)

methods = {'cea': cea, 'amtea': amtea}
# load seeds
local_config = config['ea']
seeds = range(local_config['repeat'])

cea_instances = ['nbit_6_7', 'nbit_6_8', 'nbit_6_9', 'nbit_6_10', 'nbit_8_9', 'nbit_8_10', 'nbit_8_11', 'nbit_8_12',
                 'nbit_10_11', 'nbit_10_12', 'nbit_10_13', 'nbit_10_14']
amtea_instances = ['nbit_6_10', 'nbit_8_12', 'nbit_10_14']

# cea_instances = ['nbit_6_7', 'nbit_6_8', 'nbit_8_9', 'nbit_8_10']
# amtea_instances = ['nbit_6_8', 'nbit_8_10']


# instances = ['nbit_5_8']


def amtea_ann():
    path = ''
    buildModel = False

    for seed in seeds:
        for cea_instance in cea_instances:
            if cea_instances.index(cea_instance) % 4 == 0:
                all_models = []
                path = 'all_models' + cea_instance
                Tools.save_to_file(os.path.join('problems/', path), all_models)
            if cea_instance not in amtea_instances:
                if seed == local_config['repeat'] - 1:
                    buildModel = True
                taskset = create_taskset(cea_instance)
                results = []
                # run cea
                cea(taskset, local_config, buildModel, path, callback=results.append)
                # # Logging the result to database
                method_id = get_method_id(conn, db, name='cea')
                instance_data = singletask_benchmark[cea_instance]
                hidden = instance_data['hidden']
                instance_id = get_instance_id(conn, db, cea_instance,
                                              '{}hidden'.format(' '.join('{}-'.format(hidden))))
                for result in results:
                    kwargs = {'method_id': method_id,
                              'instance_id': instance_id,
                              'best': result.fun,
                              # 'best_solution': serialize(result[k].x),
                              'num_iteration': result.nit,
                              'num_evaluation': result.nfev,
                              'seed': seed,
                              }
                    add_iteration(conn, db, **kwargs)

            if cea_instance in amtea_instances:
                buildModel = False
                taskset = create_taskset(cea_instance)
                results = []
                # run cea
                cea(taskset, local_config, buildModel, path, callback=results.append)
                # # Logging the result to database
                method_id = get_method_id(conn, db, name='cea')
                instance_data = singletask_benchmark[cea_instance]
                hidden = instance_data['hidden']
                instance_id = get_instance_id(conn, db, cea_instance,
                                              '{}hidden'.format(' '.join('{}-'.format(hidden))))
                for result in results:
                    kwargs = {'method_id': method_id,
                              'instance_id': instance_id,
                              'best': result.fun,
                              # 'best_solution': serialize(result[k].x),
                              'num_iteration': result.nit,
                              'num_evaluation': result.nfev,
                              'seed': seed,
                              }
                    add_iteration(conn, db, **kwargs)

                TrInt = 2
                trans = {'transfer': True, 'TrInt': TrInt}
                amtea(taskset, local_config, trans, False, path, callback=results.append)
                # # Logging the result to database
                method_id = get_method_id(conn, db, name='amtea')
                instance_data = singletask_benchmark[cea_instance]
                hidden = instance_data['hidden']
                instance_id = get_instance_id(conn, db, cea_instance,
                                              '{}hidden'.format(' '.join('{}-'.format(hidden))))
                for result in results:
                    kwargs = {'method_id': method_id,
                              'instance_id': instance_id,
                              'best': result.fun,
                              # 'best_solution': serialize(result[k].x),
                              'num_iteration': result.nit,
                              'num_evaluation': result.nfev,
                              'seed': seed,
                              }
                    add_iteration(conn, db, **kwargs)

            # # run amtea
            # if cea_instance in amtea_instances:
            #     TrInt = 2
            #     trans = {'transfer': True, 'TrInt': TrInt}
            #     amtea(taskset, local_config, trans, False, path, callback=results.append)
            #     # # Logging the result to database
            #     method_id = get_method_id(conn, db, name='amtea')
            #     instance_data = singletask_benchmark[cea_instance]
            #     hidden = instance_data['hidden']
            #     instance_id = get_instance_id(conn, db, cea_instance,
            #                                   '{}hidden'.format(' '.join('{}-'.format(hidden))))
            #     for result in results:
            #         kwargs = {'method_id': method_id,
            #                   'instance_id': instance_id,
            #                   'best': result.fun,
            #                   # 'best_solution': serialize(result[k].x),
            #                   'num_iteration': result.nit,
            #                   'num_evaluation': result.nfev,
            #                   'seed': seed,
            #                   }
            #         add_iteration(conn, db, **kwargs)
            #

if __name__ == "__main__":
    amtea_ann()

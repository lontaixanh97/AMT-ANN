from ann_lib import *
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

# load seeds
local_config = config['amtea']
seeds = range(local_config['repeat'])

instances = ['nbit_4_8']
TrInt = 2
trans = {'transfer': True, 'TrInt': TrInt}


def amtea_ann():
    # all_models = []
    # Tools.save_to_file(os.path.join('problems/', 'all_models'), all_models)

    for seed in seeds:
        for instance in instances:
            taskset = create_taskset(instance)
            results = []

            amtea(taskset, local_config, trans, True, callback=results.append)
            # # Logging the result to database
            method_id = get_method_id(conn, db, name='amtea')
            instance_data = singletask_benchmark[instance]
            hidden = instance_data['hidden']
            instance_id = get_instance_id(conn, db, instance,
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


if __name__ == "__main__":
    amtea_ann()

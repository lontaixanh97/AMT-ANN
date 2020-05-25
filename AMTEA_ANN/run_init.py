from cea import *
from amtea import *
from experiment import *

instances = get_config('ann_lib/singletask-benchmark.yaml')
methods = {'cea': cea, 'amtea': amtea}

config = get_config('config.yaml')
database_config = config['database']
conn = create_connection(database_config)
drop_experiment(conn, database_config)
create_experiment(conn, database_config)

cur = conn.cursor()
cur.execute('ALTER TABLE iteration ADD COLUMN rmp VARCHAR(128);')
conn.commit()

alter_method(conn, database_config)

for instance in instances:
    instance_data = instances[instance]
    hidden = instance_data['hidden']
    add_instance(conn, database_config, instance, '{}hidden'.format(' '.join('{}-'.format(hidden))))


for method in methods:
    add_method(conn, database_config, name=method)




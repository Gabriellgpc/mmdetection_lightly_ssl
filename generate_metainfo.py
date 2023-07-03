# -*- coding: utf-8 -*-
# @Author: Luis Condados
# @Date:   2023-07-02 23:59:48
# @Last Modified by:   Luis Condados
# @Last Modified time: 2023-07-03 00:09:33

import random
import click
import json
import os

def gen_random_color():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return (r, g, b)

@click.command()
@click.option('--annotation_file', '-a')
@click.option('--output', '-o')
def main(annotation_file, output):
    annotation_json = json.loads( open(annotation_file,'r').read() )

    class_names = []
    for category in annotation_json['categories']:
        class_names.append( category['name'] )

    data_metainfo = {}
    data_metainfo['classes'] = class_names
    data_metainfo['palette'] = [gen_random_color() for _ in enumerate(class_names)]
    num_classes = len(data_metainfo['classes'])

    final_output = os.path.join( './configs/dataset', output + '.py' )
    with open(final_output, 'w') as f:
        f.write('metainfo=')
        f.write(json.dumps(data_metainfo, indent=4))
        f.write(f'\nnum_classes={num_classes}')

    print('[INFO] Output saved at {}'.format(final_output))

if __name__=='__main__':
    main()
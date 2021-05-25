import ctypes
ctypes.CDLL("libpython3.7m.so.1.0")
import uuid
import io

from dearpygui.core import *
from dearpygui.simple import *
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
import scipy.stats
import networkx as nx

from collections import defaultdict, Counter


plt.style.use('dark_background')
DISTRIBUTIONS = ['norm', 'gamma', 'beta', 'cauchy', 'expon', 'halfnorm', 
    'halfcauchy', 'invgamma', 'laplace', 'lognorm', 'skewnorm', 't',
    'uniform', 'truncnorm', 'vonmises', 'wald', 'gumbel_r']
PARAM_DEFAULTS = {'loc': 0, 'scale': 1}


def link_callback(sender, data):
    print(sender, data)


def delink_callback(sender, data):
    print(sender, data)


def sample(sender, data):
    print(sender, data)
    dists = get_data('distributions')
    links = get_links('Node editor')
    node_parent = {n2: n1 for n1, n2 in links}
    print(links)

    # Construct DAG
    g = nx.DiGraph()
    for d in dists:
        g.add_node(d)
    for link in links:
        n1, n2 = [x.split('_')[0] for x in link]
        g.add_edge(n1, n2)

    # Iterate distributions from least to most number of incoming
    # connections. This take care of sample dependencies.
    node_samples = {}
    for node in nx.topological_sort(g):
        dist_name, params = dists[node]
        for param in params:
            parent_node = node_parent.get(f'{node}_{param}')
            if parent_node is None: continue
            parent_node = parent_node.split('_')[0]
            params[param] = node_samples[parent_node]
        dist =  getattr(sp.stats, dist_name)(**params)
        node_samples[node] = dist.rvs(size=1_000)

    # Update figures
    for node, samples in node_samples.items():
        print(node, samples.shape)
        fig = plt.figure(figsize=(2.5, 2))
        ax = fig.gca()
        sns.histplot(x=samples, kde=True, ax=fig.gca())
        ax.set_ylabel('')
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='raw')
        buf.seek(0)
        img_arr = np.reshape(np.frombuffer(buf.getvalue(), dtype=np.uint8),
                     newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
        buf.close()
        delete_item(f'{node}_hist', children_only=True)
        add_texture(f'{node}_texture', img_arr, img_arr.shape[1], img_arr.shape[0])
        add_image(f'{node}_plot2', f'{node}_texture', parent=f'{node}_hist')


def update_distribution(sender, data):
    name, param = data
    dists = get_data('distributions')
    dist, params = dists[name]
    params[param] = get_value(sender)
    dists[name] = (dist, params)
    add_data('distributions', dists)


def add_distribution(sender, data):
    # Add dist to data
    dist = get_value('##dists')
    params = {param: get_value(f'{param}##param') for param in get_data('params')}
    dists = get_data('distributions')
    name = f'{dist} {uuid.uuid4().hex[:6]}'
    dists[name] = (dist, params)
    add_data('distributions', dists)

    # Add node in editor
    with node(name, parent='Node editor'):
        for p, val in params.items():
            with node_attribute(f'{name}_{p}'):
                add_input_float(f'{name}_{p}_val', width=150, default_value=val, label=p,
                    callback=update_distribution, callback_data=[name, p])
        with node_attribute(f'{name}_hist', output=True):
            add_text(f'Output##{name}')


def show_dist_config(sender, data):
    dist = get_value('##dists')
    params = getattr(sp.stats, dist).shapes
    params = [] if params is None else params.split(', ')
    params = ['loc', 'scale'] + (params or [])
    add_data('params', params)

    delete_item('dist_config', children_only=True)
    for param in params:
        val = PARAM_DEFAULTS.get(param, 1)
        add_input_float(f'{param}##param', default_value=val, parent='dist_config',
            width=100)


with window("Window"):
    add_data('distributions', {}) # node name -> [dist name, params dict]

    with group('distributions', horizontal=True):
        add_text('Add distribution')
        add_combo('##dists', default_value=DISTRIBUTIONS[0], 
            items=DISTRIBUTIONS, callback=show_dist_config,
            width=200)
        with group('dist_config', horizontal=True):
            pass
        add_button('Add##dist', callback=add_distribution)

    show_dist_config('', None)

    add_button('Sample', callback=sample)

    with node_editor('Node editor', link_callback=link_callback, delink_callback=delink_callback):
        pass


# set_main_window_size(height=840, width=1100)
set_theme('Gold')
start_dearpygui(primary_window="Window")

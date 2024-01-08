# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Experiment charting script.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from absl import app
from absl import flags
import tensorflow.compat.v1.gfile as gfile

import argparse

parser = argparse.ArgumentParser(description='Active Learning Experiment Arguments')

# Add arguments to the parser
parser.add_argument('--dataset', type=str, default='wine', help='Dataset name')

args = parser.parse_args()

flags.DEFINE_string('source_dir',
                    '/home/scur1917/active-learning/toy_experiments',
                    'Directory with the output to analyze.')
flags.DEFINE_string('save_dir', '/home/scur1917/active-learning/charts',
                    'Directory to save charts.')
flags.DEFINE_string('dataset', args.dataset, 'Dataset to analyze.')
flags.DEFINE_string(
    'sampling_methods',
    ('uniform,graph_density,margin,informative_diverse,mixture_of_samplers-margin-0.33-informative_diverse-0.33-uniform-0.34'),
    'Comma separated string of sampling methods to include in chart.')
flags.DEFINE_string('scoring_methods', 'logistic,kernel_svm',
                    'Comma separated string of scoring methods to chart.')
flags.DEFINE_string('confusions', '0.0,0.2,0.4',
                    'Comma separated string of scoring methods to chart.')
flags.DEFINE_bool('normalize', False, 'Chart runs using normalized data.')
flags.DEFINE_bool('standardize', True, 'Chart runs using standardized data.')

FLAGS = flags.FLAGS


def combine_results(files, diff=False):
  all_results = {}
  for f in files:
    print("File: ", f)
    data = pickle.load(gfile.FastGFile(f, 'rb'))

    for k in data:
      if isinstance(k, tuple):
        data[k].pop('noisy_targets')
        data[k].pop('indices')
        data[k].pop('selected_inds')
        data[k].pop('sampler_output')
        key = list(k)
        seed = key[-1]
        key = key[0:10]
        key = tuple(key)
        if key in all_results:
          if seed not in all_results[key]['random_seeds']:
            all_results[key]['random_seeds'].append(seed)
            for field in [f for f in data[k] if f != 'n_points']:
              all_results[key][field] = np.vstack(
                  (all_results[key][field], data[k][field]))
        else:
          all_results[key] = data[k]
          all_results[key]['random_seeds'] = [seed]
      else:
        all_results[k] = data[k]
  return all_results


def plot_results(all_results, score_method, norm, stand, sampler_filter, confusion_filter):
  colors = {
      'margin':
          'gold',
      'uniform':
          'k',
      'informative_diverse':
          'r',
      'mixture_of_samplers-margin-0.33-informative_diverse-0.33-uniform-0.34':
          'b',
      'graph_density':
          'g'
  }
  labels = {
      'margin':
          'margin',
      'uniform':
          'uniform',
      'mixture_of_samplers-margin-0.33-informative_diverse-0.33-uniform-0.34':
          'margin:0.33,informative_diverse:0.33, uniform:0.34',
      'informative_diverse':
          'informative and diverse',
      'graph_density':
          'graph density'
  }
  markers = {
      'margin':
          'None',
      'uniform':
          'None',
      'mixture_of_samplers-margin-0.33-informative_diverse-0.33-uniform-0.34':
          '>',
      'informative_diverse':
          'None',
      'graph_density':
          'p'
  }

  fields = all_results['tuple_keys']
  fields = dict(zip(fields, range(len(fields))))

  for k in all_results.keys():
    sampler = k[fields['sampler']]
    confusion = k[fields['confusion']]

    if (isinstance(k, tuple) and
        k[fields['score_method']] == score_method and
        k[fields['standardize']] == stand and
        k[fields['normalize']] == norm and
        (sampler_filter is None or sampler in sampler_filter) and
        (confusion_filter is None or confusion == confusion_filter)):
      results = all_results[k]
      n_trials = len(results['accuracy'])
      x = results['data_sizes'][0]

      mean_acc = np.mean(results['accuracy'], axis=0)
      CI_acc = np.std(results['accuracy'], axis=0) / np.sqrt(n_trials) * 2.96
      if sampler == 'uniform':
        plt.plot(
            x,
            mean_acc,
            linewidth=1,
            label=labels[sampler],
            color=colors[sampler],
            linestyle='--'
        )
        plt.fill_between(
            x,
            mean_acc - CI_acc,
            mean_acc + CI_acc,
            color=colors[sampler],
            alpha=0.2
        )
      else:
        plt.plot(
            x,
            mean_acc,
            linewidth=1,
            label=labels[sampler],
            color=colors[sampler],
            marker=markers[sampler],
            markeredgecolor=colors[sampler]
        )
        plt.fill_between(
            x,
            mean_acc - CI_acc,
            mean_acc + CI_acc,
            color=colors[sampler],
            alpha=0.2
        )
  plt.legend(loc=4)


def get_between(filename, start, end):
  start_ind = filename.find(start) + len(start)
  end_ind = filename.rfind(end)
  return filename[start_ind:end_ind]


def get_sampling_method(dataset, filename):
  return get_between(filename, dataset + '_', '/')


def get_scoring_method(filename):
  return get_between(filename, 'results_score_', '_select_')


def get_normalize(filename):
  return get_between(filename, '_norm_', '_stand_') == 'True'


def get_standardize(filename):
  return get_between(
      filename, '_stand_', '_confusion_') == 'True'

def get_confusion(filename):
  return get_between(
      filename, '_confusion_', filename[filename.rfind('_'):])

def main(argv):
  del argv  # Unused.
  if not gfile.Exists(FLAGS.save_dir):
    gfile.MkDir(FLAGS.save_dir)
  charting_filepath = os.path.join(FLAGS.save_dir,
                                   FLAGS.dataset + '_charts.pdf')
  sampling_methods = FLAGS.sampling_methods.split(',')
  scoring_methods = FLAGS.scoring_methods.split(',')
  confusions = FLAGS.confusions.split(',')
  files = gfile.Glob(
      os.path.join(FLAGS.source_dir, FLAGS.dataset + '*/results*.pkl'))

  files = [
      f for f in files
      if (get_sampling_method(FLAGS.dataset, f) in sampling_methods and
          get_scoring_method(f) in scoring_methods and
          get_normalize(f) == FLAGS.normalize and
          get_standardize(f) == FLAGS.standardize and
          get_confusion(f) in confusions)
  ]

  print('Reading in %d files...' % len(files))
  all_results = combine_results(files)
  pdf = PdfPages(charting_filepath)

  print('Plotting charts...')
  plt.style.use('ggplot')
  for m in scoring_methods:
    for c in confusions:
      plot_results(
          all_results,
          m,
          FLAGS.normalize,
          FLAGS.standardize,
          sampler_filter=sampling_methods,
          confusion_filter=float(c))
      plt.title('Dataset: %s, Score Method: %s, Confusion: %s' % (FLAGS.dataset, m, c))
      pdf.savefig()
      plt.close()
  pdf.close()


if __name__ == '__main__':
  app.run(main)

import _init_paths
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]

from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from lib.test.evaluation import get_dataset, trackerlist

# trackers = []
# dataset_name = 'otb99_lang' # choosen from 'uav', 'nfs', 'lasot_extension_subset', 'lasot', 'otb99_lang', 'tnl2k'

# trackers.extend(trackerlist(name='seqtrack', parameter_name='seqtrack_b256', dataset_name=dataset_name,
#                             run_ids=None, display_name='seqtrack_b256'))

# dataset = get_dataset(dataset_name)

# print_results(trackers, dataset, dataset_name, merge_results=True, plot_types=('success', 'prec', 'norm_prec'),
#               force_evaluation=True)


trackers = []
dataset_name = 'sv248s_test' # choosen from 'uav', 'nfs', 'lasot_extension_subset', 'lasot'
# dataset_name = 'satsot' # choosen from 'uav', 'nfs', 'lasot_extension_subset', 'lasot'

# trackers.extend(trackerlist(name='seqtrack', parameter_name='seqtrack_l256_3d', dataset_name=dataset_name,
#                             run_ids=None, display_name='seqtrack_l256_3d'))
trackers.extend(trackerlist(name='seqtrack', parameter_name='seqtrack_b256_rmp_no_motion', dataset_name=dataset_name,
                            run_ids=None, display_name='seqtrack_b256_3d'))
# trackers.extend(trackerlist(name='seqtrack', parameter_name='seqtrack_b384_3d', dataset_name=dataset_name,
#                             run_ids=None, display_name='seqtrack_b384_3d'))

dataset = get_dataset(dataset_name)

print_results(trackers, dataset, dataset_name, merge_results=True, plot_types=('success', 'prec', 'norm_prec'),
              force_evaluation=True)


dataset = get_dataset('satsot')

print_results(trackers, dataset, 'satsot', merge_results=True, plot_types=('success', 'prec', 'norm_prec'),
              force_evaluation=True)

dataset = get_dataset('viso')

print_results(trackers, dataset, 'viso', merge_results=True, plot_types=('success', 'prec', 'norm_prec'),
              force_evaluation=True)
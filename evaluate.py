# repo originally forked from https://github.com/Confusezius/Deep-Metric-Learning-Baselines

import warnings

warnings.filterwarnings("ignore")
import numpy as np
import time
import pickle as pkl
import csv
import matplotlib.pyplot as plt

from scipy.spatial import distance
from sklearn.preprocessing import normalize

from tqdm import tqdm

import torch
import auxiliaries_nofaiss as aux

def evaluate(LOG, **kwargs):
    ret = evaluate_query_and_gallery_dataset(LOG, **kwargs)
    return ret

"""========================================================="""

class DistanceMeasure():
    """
    Container class to run and log the change of distance ratios
    between intra-class distances and inter-class distances.
    """

    def __init__(self, checkdata, opt, name='Train', update_epochs=1):
        """
        Args:
            checkdata: PyTorch DataLoader, data to check distance progression.
            opt:       argparse.Namespace, contains all training-specific parameters.
            name:      str, Name of instance. Important for savenames.
            update_epochs:  int, Only compute distance ratios every said epoch.
        Returns:
            Nothing!
        """
        self.update_epochs = update_epochs
        self.pars = opt
        self.save_path = opt.save_path

        self.name = name
        self.csv_file = opt.save_path + '/distance_measures_{}.csv'.format(self.name)
        with open(self.csv_file, 'a') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(['Rel. Intra/Inter Distance'])

        self.checkdata = checkdata

        self.mean_class_dists = []
        self.epochs = []

    def measure(self, model, epoch):
        """
        Compute distance ratios of intra- and interclass distance.
        Args:
            model: PyTorch Network, network that produces the resp. embeddings.
            epoch: Current epoch.
        Returns:
            Nothing!
        """
        if epoch % self.update_epochs: return

        self.epochs.append(epoch)

        torch.cuda.empty_cache()

        _ = model.eval()

        # Compute Embeddings
        with torch.no_grad():
            feature_coll, target_coll = [], []
            data_iter = tqdm(self.checkdata, desc='Estimating Data Distances...')
            for idx, data in enumerate(data_iter):
                input_img, target = data[1], data[0]
                features = model(input_img.to(self.pars.device))
                feature_coll.extend(features.cpu().detach().numpy().tolist())
                target_coll.extend(target.numpy().tolist())

        feature_coll = np.vstack(feature_coll).astype('float32')
        target_coll = np.hstack(target_coll).reshape(-1)
        avail_labels = np.unique(target_coll)

        # Compute indices of embeddings for each class.
        class_positions = []
        for lab in avail_labels:
            class_positions.append(np.where(target_coll == lab)[0])

        # Compute average intra-class distance and center of mass.
        com_class, dists_class = [], []
        for class_pos in class_positions:
            dists = distance.cdist(feature_coll[class_pos], feature_coll[class_pos], 'cosine')
            dists = np.sum(dists) / (len(dists) ** 2 - len(dists))
            # dists = np.linalg.norm(np.std(feature_coll_aux[class_pos],axis=0).reshape(1,-1)).reshape(-1)
            com = normalize(np.mean(feature_coll[class_pos], axis=0).reshape(1, -1)).reshape(-1)
            dists_class.append(dists)
            com_class.append(com)

        # Compute mean inter-class distances by the class-coms.
        mean_inter_dist = distance.cdist(np.array(com_class), np.array(com_class), 'cosine')
        mean_inter_dist = np.sum(mean_inter_dist) / (len(mean_inter_dist) ** 2 - len(mean_inter_dist))

        # Compute distance ratio
        mean_class_dist = np.mean(np.array(dists_class) / mean_inter_dist)
        self.mean_class_dists.append(mean_class_dist)

        self.update(mean_class_dist)

    def update(self, mean_class_dist):
        """
        Update Loggers.
        Args:
            mean_class_dist: float, Distance Ratio
        Returns:
            Nothing!
        """
        self.update_csv(mean_class_dist)
        self.update_plot()

    def update_csv(self, mean_class_dist):
        """
        Update CSV.
        Args:
            mean_class_dist: float, Distance Ratio
        Returns:
            Nothing!
        """
        with open(self.csv_file, 'a') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow([mean_class_dist])

    def update_plot(self):
        """
        Update progression plot.
        Args:
            None.
        Returns:
            Nothing!
        """
        plt.style.use('ggplot')
        f, ax = plt.subplots(1)
        ax.set_title('Mean Intra- over Interclassdistances')
        ax.plot(self.epochs, self.mean_class_dists, label='Class')
        f.legend()
        f.set_size_inches(15, 8)
        f.savefig(self.save_path + '/distance_measures_{}.svg'.format(self.name))


class GradientMeasure():
    """
    Container for gradient measure functionalities.
    Measure the gradients coming from the embedding layer to the final conv. layer
    to examine learning signal.
    """

    def __init__(self, opt, name='class-it'):
        """
        Args:
            opt:   argparse.Namespace, contains all training-specific parameters.
            name:  Name of class instance. Important for the savename.
        Returns:
            Nothing!
        """
        self.pars = opt
        self.name = name
        self.saver = {'grad_normal_mean': [], 'grad_normal_std': [], 'grad_abs_mean': [], 'grad_abs_std': []}

    def include(self, params):
        """
        Include the gradients for a set of parameters, normally the final embedding layer.
        Args:
            params: PyTorch Network layer after .backward() was called.
        Returns:
            Nothing!
        """
        gradients = [params.weight.grad.detach().cpu().numpy()]

        for grad in gradients:
            ### Shape: 128 x 2048
            self.saver['grad_normal_mean'].append(np.mean(grad, axis=0))
            self.saver['grad_normal_std'].append(np.std(grad, axis=0))
            self.saver['grad_abs_mean'].append(np.mean(np.abs(grad), axis=0))
            self.saver['grad_abs_std'].append(np.std(np.abs(grad), axis=0))

    def dump(self, epoch):
        """
        Append all gradients to a pickle file.
        Args:
            epoch: Current epoch
        Returns:
            Nothing!
        """
        with open(self.pars.save_path + '/grad_dict_{}.pkl'.format(self.name), 'ab') as f:
            pkl.dump([self.saver], f)
        self.saver = {'grad_normal_mean': [], 'grad_normal_std': [], 'grad_abs_mean': [], 'grad_abs_std': []}


"""========================================================="""


def evaluate_query_and_gallery_dataset(LOG, query_dataloader, gallery_dataloader, model, opt, save=True, give_return=False, epoch=0):
    """
    Compute evaluation metrics, update LOGGER and print results, specifically for In-Shop Clothes.
    Args:
        LOG:         aux.LOGGER-instance. Main Logging Functionality.
        query_dataloader:    PyTorch Dataloader, Query-testdata to be evaluated.
        gallery_dataloader:  PyTorch Dataloader, Gallery-testdata to be evaluated.
        model:       PyTorch Network, Network to evaluate.
        opt:         argparse.Namespace, contains all training-specific parameters.
        save:        bool, if True, Checkpoints are saved when testing metrics (specifically Recall @ 1) improve.
        give_return: bool, if True, return computed metrics.
        epoch:       int, current epoch, required for logger.
    Returns:
        (optional) Computed metrics. Are normally written directly to LOG and printed.
    """
    start = time.time()
    query_image_paths = np.array([x[0] for x in query_dataloader.dataset.image_list])
    gallery_image_paths = np.array([x[0] for x in gallery_dataloader.dataset.image_list])

    with torch.no_grad():
        # Compute Metrics.
        F1, NMI, recall_at_ks, query_feature_matrix_all, gallery_feature_matrix_all = aux.eval_metrics_query_and_gallery_dataset(model, query_dataloader, gallery_dataloader, device=opt.device, k_vals = opt.k_vals, opt=opt)
        # Generate printable summary string.
        result_str = ', '.join('@{0}: {1:.4f}'.format(k, rec) for k, rec in zip(opt.k_vals, recall_at_ks))
        result_str = 'Epoch (Test) {0}: NMI [{1:.4f}] | F1 [{2:.4f}] | Recall [{3}]'.format(epoch, NMI, F1, result_str)

        if LOG is not None:
            if save:
                aux.set_checkpoint(model, opt, LOG.progress_saver, LOG.prop.save_path+'/checkpoint.pth.tar')
                aux.recover_closest(query_feature_matrix_all, gallery_feature_matrix_all, query_image_paths, gallery_image_paths, LOG.prop.save_path+'/sample_recoveries_'+str(epoch)+'.png')
            # Update logs.
            LOG.log('val', LOG.metrics_to_log['val'], [epoch, np.round(time.time()-start), NMI, F1]+recall_at_ks)

    print(result_str)
    return recall_at_ks, NMI, F1

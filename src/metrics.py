import numpy as np
from scipy import special
from PIL import Image

from src import utils


class MeanIoU:
    """
    Calculates the mean intersection over union. `output` are expected to be logits, with
    number of channels equal to self.n_classes.
    """

    def __init__(self, n_classes, task, ignore_index=255):
        self.confusion_matrix = np.zeros((n_classes, n_classes))
        self.n_classes = n_classes
        self.task = task
        self.ignore_index = ignore_index

    def update(self, output, target, *args):
        """ Expects output and target of format (H x W x C). """

        output = np.argmax(output, axis=2)
        target = np.squeeze(target, axis=2)

        self.confusion_matrix += self._fast_hist(
            output.flatten(), target.flatten())

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true != self.ignore_index)
        hist = np.bincount(
            self.n_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.n_classes ** 2
        ).reshape(self.n_classes, self.n_classes)
        return hist

    def get_score(self, verbose=True):
        hist = self.confusion_matrix
        with np.errstate(invalid='ignore'):
            iu = np.diag(hist) / (hist.sum(axis=1) +
                                  hist.sum(axis=0) - np.diag(hist)).astype(np.float32)
        mean_iu = np.nanmean(iu)
        cls_iu = dict(zip(range(self.n_classes), iu))
        if verbose:
            print('-' * 50)
            print('Evaluation of task {}:'.format(self.task))
            print('Mean IoU is {:.4f}'.format(100 * mean_iu))
            for i in range(self.n_classes):
                print('Class IoU for class {} is {:.4f}'.format(
                    i, 100 * cls_iu[i]))
        return mean_iu

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


class ThresholdedMeanIoU:
    """
    Calculates the mean intersection over union for a binary classification problem. Optionally,
    a range of thresholds can be specified, to determine the mIoU resulting from the optimal
    classification threshold over the dataset.
    """

    def __init__(self, task, thresholds=(0.5, ), ignore_index=255):
        self.task = task
        self.thresholds = thresholds
        self.ignore_index = ignore_index
        self.jaccards = []

    def update(self, output, target, *args):
        """ Expects output and target of format (H x W x C). """

        output = special.expit(np.squeeze(output, axis=2))
        target = np.squeeze(target, axis=2)

        void_pixels = (target == self.ignore_index)
        jaccards_per_image = np.full(len(self.thresholds), np.nan)
        for idx, t in enumerate(self.thresholds):
            mask_output = (output > t)
            mask_target = (target > t)
            jaccards_per_image[idx] = self.eval_jaccard(
                mask_output, mask_target, void_pixels)
        self.jaccards.append(jaccards_per_image)

    def eval_jaccard(self, pred, gt, void_pixels):
        assert gt.shape == pred.shape
        assert void_pixels.shape == gt.shape

        gt = gt.astype(np.bool)
        pred = pred.astype(np.bool)
        void_pixels = void_pixels.astype(np.bool)
        if np.isclose(np.sum(gt & np.logical_not(void_pixels)), 0) and np.isclose(
                np.sum(pred & np.logical_not(void_pixels)), 0):
            return 1
        return np.sum(((gt & pred) & np.logical_not(void_pixels))) / \
            np.sum(((gt | pred) & np.logical_not(
                void_pixels)), dtype=np.float32)

    def get_score(self, verbose=True):
        jaccards_arr = np.array(self.jaccards)
        mean_iou = np.mean(jaccards_arr, axis=0)
        max_mean_iou = np.amax(mean_iou)
        if verbose:
            print('-' * 50)
            print('Evaluation of task {}:'.format(self.task))
            print('Max mean IoU is {:.4f}'.format(100 * max_mean_iou))
            for thresh, iou in zip(self.thresholds, mean_iou):
                print('Mean IoU for threshold {:.2f} is {:.4f}'.format(
                    thresh, 100 * iou))
        return max_mean_iou

    def reset(self):
        self.jaccards = []


class MeanErrorInAngle:
    """
    Calculates the mean error in the angles of surface normal vectors. For that purpose,
    both prediction and ground truth vectors are normalized before evaluation.
    """

    def __init__(self, task):
        self.task = task
        self.deg_diffs = []
        self.weight = []

    def update(self, output, target, *args):
        """ Expects output and target of format (H x W x C). """

        void_mask = np.bitwise_or.reduce((target == 255), axis=2)

        output = utils.normalize_array(output, dim=2)
        target = utils.normalize_array(target, dim=2)
        deg_diff_tmp = np.rad2deg(np.arccos(
            np.clip(np.sum(output * target, axis=2), a_min=-1, a_max=1)))
        res_vals = deg_diff_tmp[~void_mask]
        if len(res_vals) > 0:
            self.deg_diffs.append(np.mean(res_vals))
            self.weight.append(len(res_vals))

    def get_score(self, verbose=True):
        if len(self.deg_diffs) > 0:
            m_err = np.average(self.deg_diffs, weights=self.weight)
        else:
            m_err = None
        if verbose:
            print('-' * 50)
            print('Evaluation of task {}:'.format(self.task))
            print('Mean error in angle is {:.4f}'.format(m_err))
        return m_err

    def reset(self):
        self.deg_diffs = []
        self.weight = []


class SavePrediction:
    """
    Saves the prediction on the disk.
    """

    def __init__(self, task, save_dir):
        self.task = task
        self.save_dir = save_dir
        if task in ['edge', 'sal']:
            self.scale = lambda x: 255 * np.squeeze(special.expit(x), axis=2)
        elif task in ['semseg', 'human_parts']:
            self.scale = lambda x: np.argmax(x, axis=2)
        elif task == 'normals':
            self.scale = lambda x: 255 * \
                (utils.normalize_array(x, dim=2) + 1.0) / 2.0
        else:
            raise ValueError

    def update(self, output, target, im_size, im_name):
        """ Expects output and target of format (H x W x C). """

        scaled = self.scale(output)
        # if we used padding on the input, we crop the prediction accordingly
        if im_size != scaled.shape[:2]:
            delta_height = max(scaled.shape[0] - im_size[0], 0)
            delta_width = max(scaled.shape[1] - im_size[1], 0)
            if delta_height > 0 or delta_width > 0:
                height_location = [delta_height // 2,
                                   (delta_height // 2) + im_size[0]]
                width_location = [delta_width // 2,
                                  (delta_width // 2) + im_size[1]]
                scaled = scaled[height_location[0]:height_location[1],
                                width_location[0]:width_location[1], ...]
        assert scaled.shape[:2] == im_size
        image = Image.fromarray(scaled.astype(np.uint8))
        image.save(self.save_dir / '{}.png'.format(im_name))

    def get_score(self, verbose=True):
        if verbose:
            print('-' * 50)
            print('Evaluation of task {} is excluded.'.format(self.task))
        return str(self.save_dir)

    def reset(self):
        pass

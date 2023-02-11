import os
import sys
import json
import random
import tarfile
import cv2
import numpy as np
from PIL import Image
from scipy import io
from six.moves import urllib
from skimage import morphology
import torch
import torchvision


class PASCALContext(torch.utils.data.Dataset):
    """
    Taken from https://github.com/facebookresearch/astmt

    PASCAL-Context dataset, for multiple tasks
    Included tasks:
        1. Edge detection,
        2. Semantic Segmentation,
        3. Human Part Segmentation,
        4. Surface Normal prediction (distilled),
        5. Saliency (distilled)
    """

    URL = 'https://data.vision.ee.ethz.ch/kmaninis/share/MTL/PASCAL_MT.tgz'

    HUMAN_PART = {1: {'hair': 1, 'head': 1, 'lear': 1, 'lebrow': 1, 'leye': 1, 'lfoot': 1,
                      'lhand': 1, 'llarm': 1, 'llleg': 1, 'luarm': 1, 'luleg': 1, 'mouth': 1,
                      'neck': 1, 'nose': 1, 'rear': 1, 'rebrow': 1, 'reye': 1, 'rfoot': 1,
                      'rhand': 1, 'rlarm': 1, 'rlleg': 1, 'ruarm': 1, 'ruleg': 1, 'torso': 1},
                  4: {'hair': 1, 'head': 1, 'lear': 1, 'lebrow': 1, 'leye': 1, 'lfoot': 4,
                      'lhand': 3, 'llarm': 3, 'llleg': 4, 'luarm': 3, 'luleg': 4, 'mouth': 1,
                      'neck': 2, 'nose': 1, 'rear': 1, 'rebrow': 1, 'reye': 1, 'rfoot': 4,
                      'rhand': 3, 'rlarm': 3, 'rlleg': 4, 'ruarm': 3, 'ruleg': 4, 'torso': 2},
                  6: {'hair': 1, 'head': 1, 'lear': 1, 'lebrow': 1, 'leye': 1, 'lfoot': 6,
                      'lhand': 4, 'llarm': 4, 'llleg': 6, 'luarm': 3, 'luleg': 5, 'mouth': 1,
                      'neck': 2, 'nose': 1, 'rear': 1, 'rebrow': 1, 'reye': 1, 'rfoot': 6,
                      'rhand': 4, 'rlarm': 4, 'rlleg': 6, 'ruarm': 3, 'ruleg': 5, 'torso': 2},
                  14: {'hair': 1, 'head': 1, 'lear': 1, 'lebrow': 1, 'leye': 1, 'lfoot': 14,
                       'lhand': 8, 'llarm': 7, 'llleg': 13, 'luarm': 6, 'luleg': 12, 'mouth': 1,
                       'neck': 2, 'nose': 1, 'rear': 1, 'rebrow': 1, 'reye': 1, 'rfoot': 11,
                       'rhand': 5, 'rlarm': 4, 'rlleg': 10, 'ruarm': 3, 'ruleg': 9, 'torso': 2}
                  }

    VOC_CATEGORY_NAMES = ['background',
                          'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                          'bus', 'car', 'cat', 'chair', 'cow',
                          'diningtable', 'dog', 'horse', 'motorbike', 'person',
                          'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

    CONTEXT_CATEGORY_LABELS = [0,
                               2, 23, 25, 31, 34,
                               45, 59, 65, 72, 98,
                               397, 113, 207, 258, 284,
                               308, 347, 368, 416, 427]

    def __init__(self,
                 data_dir,
                 download=True,
                 split='val',
                 transforms=False,
                 area_thres=0,
                 retname=True,
                 overfit=False,
                 tasks=('semseg',),
                 num_human_parts=6,
                 use_resized=False):

        if download:
            self._download(data_dir, use_resized)

        if use_resized:
            self.root = os.path.join(data_dir, 'PASCAL_MT', 'resized')
        else:
            self.root = os.path.join(data_dir, 'PASCAL_MT')

        image_dir = os.path.join(self.root, 'JPEGImages')

        if transforms:
            self.transforms = self._get_transforms(split, use_resized)
        else:
            self.transforms = None

        assert isinstance(split, str)
        self.split = [split]

        self.area_thres = area_thres
        self.retname = retname

        # Edge Detection
        self.do_edge = ('edge' in tasks)
        self.edges = []
        edge_gt_dir = os.path.join(self.root, 'pascal-context', 'trainval')

        # Semantic Segmentation
        self.do_semseg = ('semseg' in tasks)
        self.semsegs = []

        # Human Part Segmentation
        self.do_human_parts = ('human_parts' in tasks)
        part_gt_dir = os.path.join(self.root, 'human_parts')
        self.parts = []
        self.human_parts_category = 15
        self.cat_part = json.load(open(os.path.join(os.path.dirname(__file__),
                                                    'db_info/pascal_part.json'), 'r'))
        self.cat_part["15"] = self.HUMAN_PART[num_human_parts]
        self.parts_file = os.path.join(self.root, 'ImageSets', 'Parts',
                                       ''.join(self.split) + '.txt')

        # Surface Normal Estimation
        self.do_normals = ('normals' in tasks)
        _normal_gt_dir = os.path.join(self.root, 'normals_distill')
        self.normals = []
        if self.do_normals:
            with open(os.path.join(os.path.dirname(__file__), 'db_info/nyu_classes.json')) as f:
                cls_nyu = json.load(f)
            with open(os.path.join(os.path.dirname(__file__), 'db_info/context_classes.json')) as f:
                cls_context = json.load(f)

            self.normals_valid_classes = []
            for cl_nyu in cls_nyu:
                if cl_nyu in cls_context and cl_nyu != 'unknown':
                    self.normals_valid_classes.append(cls_context[cl_nyu])

            # Custom additions due to incompatibilities
            self.normals_valid_classes.append(cls_context['tvmonitor'])

        # Saliency
        self.do_sal = ('sal' in tasks)
        _sal_gt_dir = os.path.join(self.root, 'sal_distill')
        self.sals = []

        # train/val/test splits are pre-cut
        _splits_dir = os.path.join(self.root, 'ImageSets', 'Context')

        self.im_ids = []
        self.images = []

        for splt in self.split:
            with open(os.path.join(_splits_dir, splt + '.txt'), "r") as f:
                lines = f.read().splitlines()

            for ii, line in enumerate(lines):
                # Images
                if use_resized:
                    _image = os.path.join(image_dir, line + ".png")
                else:
                    _image = os.path.join(image_dir, line + ".jpg")
                assert os.path.isfile(_image), _image
                self.images.append(_image)
                self.im_ids.append(line.rstrip('\n'))

                # Edges
                _edge = os.path.join(edge_gt_dir, line + ".mat")
                assert os.path.isfile(_edge), _edge
                self.edges.append(_edge)

                # Semantic Segmentation
                _semseg = self._get_semseg_fname(line)
                assert os.path.isfile(_semseg), _semseg
                self.semsegs.append(_semseg)

                # Human Parts
                _human_part = os.path.join(part_gt_dir, line + ".mat")
                assert os.path.isfile(_human_part), _human_part
                self.parts.append(_human_part)

                _normal = os.path.join(_normal_gt_dir, line + ".png")
                assert os.path.isfile(_normal), _normal
                self.normals.append(_normal)

                _sal = os.path.join(_sal_gt_dir, line + ".png")
                assert os.path.isfile(_sal), _sal
                self.sals.append(_sal)

        if self.do_edge:
            assert len(self.images) == len(self.edges)
        if self.do_human_parts:
            assert len(self.images) == len(self.parts)
        if self.do_semseg:
            assert len(self.images) == len(self.semsegs)
        if self.do_normals:
            assert len(self.images) == len(self.normals)
        if self.do_sal:
            assert len(self.images) == len(self.sals)

        if not self._check_preprocess_parts():
            self._preprocess_parts()

        if self.do_human_parts:
            # Find images which have human parts
            self.has_human_parts = []
            for ii in range(len(self.im_ids)):
                if self.human_parts_category in self.part_obj_dict[self.im_ids[ii]]:
                    self.has_human_parts.append(1)
                else:
                    self.has_human_parts.append(0)

            # If the other tasks are disabled, select only the images that contain human parts,
            # to allow batching
            if not self.do_edge and not self.do_semseg and not self.do_sal and not self.do_normals:
                for i in range(len(self.parts) - 1, -1, -1):
                    if self.has_human_parts[i] == 0:
                        del self.im_ids[i]
                        del self.images[i]
                        del self.parts[i]
                        del self.has_human_parts[i]

        #  Overfit to n_of images
        if overfit:
            n_of = 64
            self.images = self.images[:n_of]
            self.im_ids = self.im_ids[:n_of]
            if self.do_edge:
                self.edges = self.edges[:n_of]
            if self.do_semseg:
                self.semsegs = self.semsegs[:n_of]
            if self.do_human_parts:
                self.parts = self.parts[:n_of]
            if self.do_normals:
                self.normals = self.normals[:n_of]
            if self.do_sal:
                self.sals = self.sals[:n_of]

    def __getitem__(self, index):
        sample = {}

        _img, lab_shape = self._load_img(index)
        sample['image'] = _img

        if self.do_edge:
            _edge = self._load_edge(index)
            assert _edge.shape == lab_shape
            sample['edge'] = _edge

        if self.do_human_parts:
            _human_parts, _ = self._load_human_parts(index, lab_shape)
            assert _human_parts.shape == lab_shape
            sample['human_parts'] = _human_parts

        if self.do_semseg:
            _semseg = self._load_semseg(index)
            assert _semseg.shape == lab_shape
            sample['semseg'] = _semseg

        if self.do_normals:
            _normals = self._load_normals_distilled(index)
            assert _normals.shape[:2] == lab_shape
            sample['normals'] = _normals

        if self.do_sal:
            _sal = self._load_sal_distilled(index)
            assert _sal.shape[:2] == lab_shape
            sample['sal'] = _sal

        if self.retname:
            sample['meta'] = {'image': str(self.im_ids[index]),
                              'im_size': (_img.shape[0], _img.shape[1])}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self):
        return len(self.images)

    def _load_img(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        lab_shape = _img.size[::-1]
        _img = np.array(_img, dtype=np.float32)
        return _img, lab_shape

    def _load_edge(self, index):
        # Read Target object
        _tmp = io.loadmat(self.edges[index])['LabelMap']
        _edge = cv2.Laplacian(_tmp, cv2.CV_64F)
        _edge = morphology.thin(np.abs(_edge) > 0).astype(np.float32)
        return _edge

    def _load_human_parts(self, index, lab_shape):
        if self.has_human_parts[index]:

            # Read Target object
            _part_mat = io.loadmat(self.parts[index])['anno'][0][0][1][0]
            _inst_mask = _target = None

            for _obj in _part_mat:

                has_human = _obj[1][0][0] == self.human_parts_category
                has_parts = len(_obj[3]) != 0

                if has_human and has_parts:
                    _inter = _obj[2].astype(np.float32)
                    if _inst_mask is None:
                        _inst_mask = _inter
                        _target = np.zeros(_inst_mask.shape)
                    else:
                        _inst_mask = np.maximum(_inst_mask, _inter)

                    n_parts = len(_obj[3][0])
                    for part_i in range(n_parts):
                        cat_part = str(_obj[3][0][part_i][0][0])
                        mask_id = self.cat_part[str(
                            self.human_parts_category)][cat_part]
                        mask = _obj[3][0][part_i][1].astype(bool)
                        _target[mask] = mask_id

            if _target is not None:
                _target, _inst_mask = _target.astype(
                    np.float32), _inst_mask.astype(np.float32)
            else:
                _target, _inst_mask = np.zeros(lab_shape, dtype=np.float32), np.zeros(
                    lab_shape, dtype=np.float32)
            return _target, _inst_mask

        return np.zeros(lab_shape, dtype=np.float32), np.zeros(lab_shape, dtype=np.float32)

    def _load_semseg(self, index):
        _semseg = Image.open(self.semsegs[index])
        _semseg = np.array(_semseg, dtype=np.float32)
        return _semseg

    def _load_normals_distilled(self, index):
        _tmp = Image.open(self.normals[index])
        _tmp = np.array(_tmp, dtype=np.float32)
        _tmp = 2.0 * _tmp / 255.0 - 1.0

        labels = io.loadmat(os.path.join(
            self.root, 'pascal-context', 'trainval', self.im_ids[index] + '.mat'))
        labels = labels['LabelMap']
        _normals = np.zeros(_tmp.shape, dtype=np.float)
        for x in np.unique(labels):
            if x in self.normals_valid_classes:
                _normals[labels == x, :] = _tmp[labels == x, :]
        return _normals

    def _load_sal_distilled(self, index):
        _sal = Image.open(self.sals[index])
        _sal = np.array(_sal, dtype=np.float32) / 255.
        _sal = (_sal > 0.5).astype(np.float32)
        return _sal

    def _get_semseg_fname(self, fname):
        fname_voc = os.path.join(self.root, 'semseg', 'VOC12', fname + '.png')
        fname_context = os.path.join(
            self.root, 'semseg', 'pascal-context', fname + '.png')
        if os.path.isfile(fname_voc):
            seg = fname_voc
        elif os.path.isfile(fname_context):
            seg = fname_context
        else:
            seg = None
        return seg

    @staticmethod
    def _get_transforms(split, use_resized):
        if use_resized:
            size = [256, 256]
        else:
            size = [512, 512]
        if split == 'train':
            transform = torchvision.transforms.Compose([
                RandomScaling(min_scale_factor=0.5,
                              max_scale_factor=2.0, step_size=0.25),
                RandomCrop(size=size),
                RandomHorizontalFlip(),
                AddIgnoreRegions(),
                ToTensor(),
                ZeroMeanUnitRange()
            ])
        else:
            transform = torchvision.transforms.Compose([
                PadImage(size=size),
                AddIgnoreRegions(),
                ToTensor(),
                ZeroMeanUnitRange()
            ])
        return transform

    def _check_preprocess_parts(self):
        _obj_list_file = self.parts_file
        if not os.path.isfile(_obj_list_file):
            return False
        self.part_obj_dict = json.load(open(_obj_list_file, 'r'))
        return list(np.sort([str(x) for x in self.part_obj_dict.keys()])) \
            == list(np.sort(self.im_ids))

    def _preprocess_parts(self):
        self.part_obj_dict = {}
        obj_counter = 0
        for ii in range(len(self.im_ids)):
            # Read object masks and get number of objects
            part_mat = io.loadmat(
                os.path.join(self.root, 'human_parts', '{}.mat'.format(self.im_ids[ii])))
            n_obj = len(part_mat['anno'][0][0][1][0])

            # Get the categories from these objects
            _cat_ids = []
            for jj in range(n_obj):
                obj_area = np.sum(part_mat['anno'][0][0][1][0][jj][2])
                if obj_area > self.area_thres:
                    _cat_ids.append(int(part_mat['anno'][0][0][1][0][jj][1]))
                else:
                    _cat_ids.append(-1)
                obj_counter += 1

            self.part_obj_dict[self.im_ids[ii]] = _cat_ids

        with open(self.parts_file, 'w') as outfile:
            outfile.write('{{\n\t"{:s}": {:s}'.format(
                self.im_ids[0], json.dumps(self.part_obj_dict[self.im_ids[0]])))
            for ii in range(1, len(self.im_ids)):
                outfile.write(',\n\t"{:s}": {:s}'.format(
                    self.im_ids[ii], json.dumps(self.part_obj_dict[self.im_ids[ii]])))
            outfile.write('\n}\n')

    def _download(self, data_dir, use_resized):

        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        _fpath = os.path.join(data_dir, 'PASCAL_MT.tgz')

        if not os.path.isfile(_fpath):
            def _progress(count, block_size, total_size):
                sys.stdout.write('\r>> %s %.1f%%' %
                                 (_fpath, float(count * block_size)
                                  / float(total_size) * 100.0))
                sys.stdout.flush()

            urllib.request.urlretrieve(self.URL, _fpath, _progress)
            print('\n')
            print('Extracting dataset...')

            # extract file
            cwd = os.getcwd()
            tar = tarfile.open(_fpath)
            os.chdir(data_dir)
            tar.extractall(path=data_dir)
            tar.close()
            os.chdir(cwd)

            # automatically generate resized version of the dataset
            self.resize_dataset(os.path.join(data_dir, 'PASCAL_MT'))

    def resize_dataset(self, dataset_dir):

        print('Resizing dataset, this could take a while...')

        def recursive_glob(rootdir='.', suffix=''):
            return [os.path.join(looproot, filename)
                    for looproot, _, filenames in os.walk(rootdir)
                    for filename in filenames if filename.endswith(suffix)]

        filepaths = recursive_glob(rootdir=dataset_dir, suffix='.png')

        assert len(filepaths) == 33224, len(filepaths)

        for i, f in enumerate(filepaths):
            image = Image.open(f)
            if 'normals_distill' in f:
                res = Image.LANCZOS
            elif 'sal_distill' in f:
                res = Image.LANCZOS
            elif 'semseg' in f:
                res = Image.NEAREST
            else:
                raise ValueError
            new_size = [el // 2 for el in image.size]
            image = image.resize(new_size, resample=res)
            new_f = 'PASCAL_MT/resized'.join(f.rsplit('PASCAL_MT', 1))
            if not os.path.exists(os.path.dirname(new_f)):
                os.makedirs(os.path.dirname(new_f))
            image.save(new_f)

            if i % 1000 == 0:
                print('{} / 33224 .png files resized'.format(i))

        # we save .jpg images as .png since we want to avoid compression
        filepaths = recursive_glob(rootdir=dataset_dir, suffix='.jpg')

        assert len(filepaths) == 10103, len(filepaths)

        for i, f in enumerate(filepaths):
            image = Image.open(f)
            if 'JPEGImages' in f:
                res = Image.LANCZOS
            else:
                raise ValueError
            new_size = [el // 2 for el in image.size]
            image = image.resize(new_size, resample=res)
            new_f = 'PASCAL_MT/resized'.join(f.rsplit('PASCAL_MT', 1))
            # JPEG images might be compressed badly
            new_f = new_f.replace('.jpg', '.png')
            if not os.path.exists(os.path.dirname(new_f)):
                os.makedirs(os.path.dirname(new_f))
            image.save(new_f)

            if i % 1000 == 0:
                print('{} / 10103 .jpg resized'.format(i))

        # need to do the same for .mat files
        filepaths = recursive_glob(rootdir=dataset_dir, suffix='.mat')

        assert len(filepaths) == 20206, len(filepaths)

        for i, f in enumerate(filepaths):
            matfile = io.loadmat(f)
            if 'pascal-context' in f:
                image = matfile['LabelMap']
                new_size = tuple([el // 2 for el in image.shape[::-1]])
                image = cv2.resize(image, dsize=new_size,
                                   interpolation=cv2.INTER_NEAREST)
                matfile['LabelMap'] = image.astype(np.uint16)
            elif 'human_parts' in f:
                part_mat = matfile['anno'][0][0][1][0]
                for j, arr in enumerate(part_mat):
                    inst_mask = arr[2]
                    new_size = tuple([el // 2 for el in inst_mask.shape[::-1]])
                    inst_mask = cv2.resize(
                        inst_mask, dsize=new_size, interpolation=cv2.INTER_NEAREST)
                    part_mat[j][2] = inst_mask.astype(np.uint8)
                    if len(arr[3]) > 0:
                        for k, arrarr in enumerate(arr[3][0]):
                            mask = arrarr[1]
                            new_size = tuple(
                                [el // 2 for el in mask.shape[::-1]])
                            mask = cv2.resize(mask, dsize=new_size,
                                              interpolation=cv2.INTER_NEAREST)
                            part_mat[j][3][0][k][1] = mask.astype(np.uint8)
            else:
                raise ValueError
            new_f = 'PASCAL_MT/resized'.join(f.rsplit('PASCAL_MT', 1))
            if not os.path.exists(os.path.dirname(new_f)):
                os.makedirs(os.path.dirname(new_f))
            io.savemat(new_f, matfile, do_compression=True)

            if i % 1000 == 0:
                print('{} / 20206 .mat resized'.format(i))

        txt_files = recursive_glob(rootdir=dataset_dir, suffix='.txt')
        for f in txt_files:
            new_f = 'PASCAL_MT/resized'.join(f.rsplit('PASCAL_MT', 1))
            if not os.path.exists(os.path.dirname(new_f)):
                os.makedirs(os.path.dirname(new_f))
            os.system('cp {} {}'.format(f, new_f))

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RandomHorizontalFlip:
    """Horizontally flip the given image and ground truth randomly."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):

        if random.random() < self.p:
            for elem in sample.keys():
                if 'meta' in elem:
                    continue

                tmp = sample[elem]
                tmp = cv2.flip(tmp, flipCode=1)
                sample[elem] = tmp

                if elem == 'normals':
                    sample[elem][:, :, 0] *= -1

        return sample

    def __repr__(self):
        return self.__class__.__name__ + '()'


class AddIgnoreRegions:
    """Add Ignore Regions"""

    def __call__(self, sample):

        for elem in sample.keys():
            tmp = sample[elem]

            if elem == 'normals':
                # Check areas with norm 0
                norm = np.sqrt(tmp[:, :, 0] ** 2 + tmp[:, :, 1]
                               ** 2 + tmp[:, :, 2] ** 2)
                tmp[norm == 0, :] = 255
                sample[elem] = tmp
            elif elem == 'human_parts':
                # Check for images without human part annotations
                if ((tmp == 0) | (tmp == 255)).all():
                    tmp = np.full(tmp.shape, 255, dtype=tmp.dtype)
                    sample[elem] = tmp
            elif elem == 'depth':
                tmp[tmp == 0] = 255
                sample[elem] = tmp

        return sample

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ToTensor:
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):

        for elem in sample.keys():
            if 'meta' in elem:
                continue
            elif 'bbox' in elem:
                tmp = sample[elem]
                sample[elem] = torch.from_numpy(tmp)
                continue

            tmp = sample[elem]

            if tmp.ndim == 2:
                tmp = tmp[:, :, None]
            img = torch.from_numpy(tmp.transpose((2, 0, 1))).float()
            sample[elem] = img

        return sample

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ZeroMeanUnitRange:
    """ Map image values from [0, 255] to [-1, 1].
    As for instance done for MobileNetV2 in official TF repo:
    https://github.com/tensorflow/models/blob/master/research/deeplab/core/feature_extractor.py#L466
    """

    def __call__(self, sample):
        tmp = sample['image']
        sample['image'] = (2.0 / 255.0) * tmp.float() - 1.0
        return sample

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RandomScaling:
    """Random scale the input.
    Args:
      min_scale_factor: Minimum scale value.
      max_scale_factor: Maximum scale value.
      step_size: The step size from minimum to maximum value.
    Returns:
        sample: The input sample scaled
    """

    def __init__(self, min_scale_factor=1.0, max_scale_factor=1.0, step_size=0):
        if min_scale_factor < 0 or min_scale_factor > max_scale_factor:
            raise ValueError('Unexpected value of min_scale_factor')
        self.min_scale_factor = float(min_scale_factor)
        self.max_scale_factor = float(max_scale_factor)
        self.step_size = step_size

    def get_random_scale(self):
        """Gets a random scaling value.
        Args:
            min_scale_factor: Minimum scale value.
            max_scale_factor: Maximum scale value.
            step_size: The step size from minimum to maximum value.
        Returns:
            A random scale value selected between minimum and maximum value.
        """

        if self.min_scale_factor == self.max_scale_factor:
            rand_scale = self.min_scale_factor

        # Uniformly sampling of the value from [min, max) when step_size = 0
        elif self.step_size == 0:
            rand_scale = random.uniform(
                self.min_scale_factor, self.max_scale_factor)
        # Else, randomly select one discrete value from [min, max]
        else:
            num_steps = int(
                (self.max_scale_factor - self.min_scale_factor) / self.step_size)
            rand_step = random.randint(0, num_steps)
            rand_scale = self.min_scale_factor + rand_step * self.step_size
        return rand_scale

    def scale(self, key, unscaled, scale=1.0):
        """Randomly scales image and label.
        Args:
            key: Key indicating the uscaled input origin
            unscaled: Image or target to be scaled.
            scale: The value to scale image and label.
        Returns:
            scaled: The scaled image or target
        """
        # No random scaling if scale == 1.
        if scale == 1.0:
            return unscaled
        image_shape = np.shape(unscaled)[0:2]
        new_dim = tuple([int(x * scale) for x in image_shape])

        if key in {'image'}:
            scaled = cv2.resize(
                unscaled, new_dim[::-1], interpolation=cv2.INTER_LINEAR)
        elif key in {'semseg', 'human_parts', 'edge', 'sal', 'normals', 'depth'}:
            scaled = cv2.resize(
                unscaled, new_dim[::-1], interpolation=cv2.INTER_NEAREST)
        else:
            raise ValueError(
                'Key {} for input origin is not supported'.format(key))

        if key in {'depth'}:
            # ignore regions for depth are 0
            scaled /= scale

        return scaled

    def __call__(self, sample):

        random_scale = self.get_random_scale()
        for key, val in sample.items():
            if 'meta' in key:
                continue
            sample[key] = self.scale(key, val, scale=random_scale)
        return sample

    def __repr__(self):
        return self.__class__.__name__ + '()'


class PadImage:
    """Pad image and label to have dimensions >= [size_height, size_width]
    Args:
        size: Desired size
    Returns:
        sample: The input sample padded
    """

    def __init__(self, size, image_fill_index=(127.5, 127.5, 127.5)):
        if isinstance(size, int):
            self.size = tuple([size, size])
        elif isinstance(size, list):
            self.size = tuple(size)
        else:
            raise ValueError('Crop size must be an int or a list')

        self.fill_index = {'edge': 255,
                           'human_parts': 255,
                           'semseg': 255,
                           'depth': 0,
                           'normals': [0, 0, 0],
                           'sal': 255,
                           'image': image_fill_index}

    def pad(self, key, unpadded):
        unpadded_shape = np.shape(unpadded)
        delta_height = max(self.size[0] - unpadded_shape[0], 0)
        delta_width = max(self.size[1] - unpadded_shape[1], 0)

        if delta_height == 0 and delta_width == 0:
            return unpadded

        # Location to place image
        height_location = [delta_height // 2,
                           (delta_height // 2) + unpadded_shape[0]]
        width_location = [delta_width // 2,
                          (delta_width // 2) + unpadded_shape[1]]

        pad_value = self.fill_index[key]
        max_height = max(self.size[0], unpadded_shape[0])
        max_width = max(self.size[1], unpadded_shape[1])
        if key in {'image', 'normals'}:
            padded = np.full((max_height, max_width, 3),
                             pad_value, dtype=np.float32)
            padded[height_location[0]:height_location[1],
                   width_location[0]:width_location[1], :] = unpadded
        elif key in {'semseg', 'human_parts', 'edge', 'sal', 'depth'}:
            padded = np.full((max_height, max_width),
                             pad_value, dtype=np.float32)
            padded[height_location[0]:height_location[1],
                   width_location[0]:width_location[1]] = unpadded
        else:
            raise ValueError(
                'Key {} for input origin is not supported'.format(key))
        return padded

    def __call__(self, sample):
        for key, val in sample.items():
            if 'meta' in key:
                continue
            sample[key] = self.pad(key, val)
        return sample

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RandomCrop:
    """Random crop image if it exceeds desired size
    Args:
        size: Desired size
    Returns:
        sample: The input sample randomly cropped
    """

    def __init__(self, size, image_fill_index=(127.5, 127.5, 127.5)):
        if isinstance(size, int):
            self.size = tuple([size, size])
        elif isinstance(size, list):
            self.size = tuple(size)
        else:
            raise ValueError('Crop size must be an int or a list')
        self.padding = PadImage(size, image_fill_index)

    def get_random_crop_loc(self, uncropped):
        """Gets a random crop location.
        Args:
            key: Key indicating the uncropped input origin
            uncropped: Image or target to be cropped.
        Returns:
            Cropping region.
        """
        uncropped_shape = np.shape(uncropped)
        img_height = uncropped_shape[0]
        img_width = uncropped_shape[1]

        desired_height = self.size[0]
        desired_width = self.size[1]
        if img_height == desired_height and img_width == desired_width:
            return None
        # Get random offset uniformly from [0, max_offset]
        max_offset_height = img_height - desired_height
        max_offset_width = img_width - desired_width

        offset_height = random.randint(0, max_offset_height)
        offset_width = random.randint(0, max_offset_width)
        crop_loc = np.array([[offset_height, offset_height + desired_height],
                             [offset_width, offset_width + desired_width]])
        return crop_loc

    def random_crop(self, key, uncropped, crop_loc):
        if crop_loc is None:
            return uncropped
        if key in {'image', 'normals'}:
            cropped = uncropped[crop_loc[0, 0]:crop_loc[0, 1],
                                crop_loc[1, 0]:crop_loc[1, 1], :]
        elif key in {'semseg', 'human_parts', 'edge', 'sal', 'depth'}:
            cropped = uncropped[crop_loc[0, 0]:crop_loc[0, 1],
                                crop_loc[1, 0]:crop_loc[1, 1]]
        else:
            raise ValueError(
                'Key {} for input origin is not supported'.format(key))
        assert np.shape(cropped)[0:2] == self.size
        return cropped

    def __call__(self, sample):
        # Ensure the image is at least as large as the desired size
        sample = self.padding(sample)
        crop_location = self.get_random_crop_loc(sample['image'])
        for key, val in sample.items():
            if 'meta' in key:
                continue
            sample[key] = self.random_crop(key, val, crop_location)
        return sample

    def __repr__(self):
        return self.__class__.__name__ + '()'

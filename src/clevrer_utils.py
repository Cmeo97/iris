"""
Credits to https://github.com/Wuziyi616/nerv & https://github.com/pairlab/SlotFormer
"""

import json, os
from collections import OrderedDict
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
import cv2
from cv2 import (CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FPS,
                 CAP_PROP_FRAME_COUNT, CAP_PROP_FOURCC, CAP_PROP_POS_FRAMES,
                 VideoWriter_fourcc)
import pickle
import six
import numpy as np
import torch
from torchvision.ops import masks_to_boxes
import torchvision.transforms as transforms
from torchvision.transforms import functional as F

import pycocotools.mask as mask_utils

def json_load(file):
    if isinstance(file, str):
        with open(file, 'r') as f:
            obj = json.load(f)
    elif hasattr(file, 'read'):
        obj = json.load(file)
    else:
        raise TypeError('"file" must be a filename str or a file-object')
    return obj

def yaml_load(file, **kwargs):
    kwargs.setdefault('Loader', Loader)
    if isinstance(file, str):
        with open(file, 'r') as f:
            obj = yaml.load(f, **kwargs)
    elif hasattr(file, 'read'):
        obj = yaml.load(file, **kwargs)
    else:
        raise TypeError('"file" must be a filename str or a file-object')
    return obj

def pickle_load(file, **kwargs):
    if isinstance(file, str):
        with open(file, 'rb') as f:
            obj = pickle.load(f, **kwargs)
    elif hasattr(file, 'read'):
        obj = pickle.load(file, **kwargs)
    else:
        raise TypeError('"file" must be a filename str or a file-object')
    return obj

def check_file_exist(filename, msg_tmpl='file "{}" not exist:'):
    """Check whether a file exists."""
    if not os.path.isfile(filename):
        raise FileNotFoundError(msg_tmpl.format(filename))
    
def mkdir_or_exist(dir_name):
    """Create a new directory if not existed."""
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)

def strip_suffix(file):
    """Return the filename without suffix.
    E.g. 'xxx/video' for 'xxx/video.mp4'.

    Args:
        file (str): string to be processed.

    Returns:
        str: filename without suffix.
    """
    assert isinstance(file, str)
    suffix = file.split('.')[-1]
    return file[:-(len(suffix) + 1)]

def load_obj(file, format=None, **kwargs):
    """Load contents from json/yaml/pickle files, and also supports custom
    arguments for each file format.

    This method provides a unified api for loading from serialized files.

    Args:
        file (str or file-like object): filename or the file-like object.
        format (None or str): if it is None, file format is inferred from the
            file extension, otherwise use the specified one. Currently
            supported formats are "json", "yaml", "yml", "pickle" and "pkl".

    Returns:
        Any: The content from the file.
    """
    processors = {
        'json': json_load,
        'yaml': yaml_load,
        'yml': yaml_load,
        'pickle': pickle_load,
        'pkl': pickle_load
    }
    if format is None and isinstance(file, str):
        format = file.split('.')[-1]
    if format not in processors:
        raise TypeError('Unsupported format: ' + format)
    return processors[format](file, **kwargs)

def read_img(img_or_path, flag=cv2.IMREAD_COLOR):
    """Read an image.

    Args:
        img_or_path (np.ndarray or str): either an image or path of an image.
        flag (int, optional): flags specifying the color type of loaded image.
            Default: cv2.IMREAD_COLOR.

    Returns:
        np.ndarray: loaded image array.
    """
    if isinstance(img_or_path, np.ndarray):
        return img_or_path
    elif isinstance(img_or_path, six.string_types):
        check_file_exist(img_or_path,
                         'img file does not exist: {}'.format(img_or_path))
        return cv2.imread(img_or_path, flag)
    else:
        raise TypeError('"img" must be a numpy array or a filename')


class Cache(object):

    def __init__(self, capacity):
        self._cache = OrderedDict()
        self._capacity = int(capacity)
        if capacity <= 0:
            raise ValueError('capacity must be a positive integer')

    @property
    def capacity(self):
        return self._capacity

    @property
    def size(self):
        return len(self._cache)

    def put(self, key, val):
        if key in self._cache:
            return
        if len(self._cache) >= self.capacity:
            self._cache.popitem(last=False)
        self._cache[key] = val

    def get(self, key, default=None):
        val = self._cache[key] if key in self._cache else default
        return val

def resize(img, size, return_scale=False, interpolation=cv2.INTER_LINEAR):
    """Resize image by expected size

    Args:
        img (np.ndarray): image or image path.
        size (Tuple[int]): (w, h).
        return_scale (bool, optional): whether to return w_scale and h_scale.
            Default: False.
        interpolation (enum, optional): interpolation method.
            Default: cv2.INTER_LINEAR.

    Returns:
        np.ndarray: resized image.
    """
    img = read_img(img)
    h, w = img.shape[:2]
    resized_img = cv2.resize(img, size, interpolation=interpolation)
    if not return_scale:
        return resized_img
    else:
        w_scale = size[0] / float(w)
        h_scale = size[1] / float(h)
        return resized_img, w_scale, h_scale

class VideoReader(object):
    """Video class with similar usage to a list object.

    This video warpper class provides convenient apis to access frames.
    There exists an issue of OpenCV's VideoCapture class that jumping to a
    certain frame may be inaccurate. It is fixed in this class by checking
    the position after jumping each time.

    Cache is used when decoding videos. So if the same frame is visited for the
    second time, there is no need to decode again if it is stored in the cache.

    :Example:

    >>> import cvbase as cvb
    >>> v = cvb.VideoReader('sample.mp4')
    >>> len(v)  # get the total frame number with `len()`
    120
    >>> for img in v:  # v is iterable
    >>>     cvb.show_img(img)
    >>> v[5]  # get the 6th frame

    """

    def __init__(self, filename, cache_capacity=-1, to_rgb=False):
        check_file_exist(filename, 'Video file not found: ' + filename)
        self.filename = filename
        self._vcap = cv2.VideoCapture(filename)
        self._cache = Cache(cache_capacity) if cache_capacity > 0 else None
        self._to_rgb = to_rgb  # convert img from GBR to RGB when reading
        self._position = 0
        # get basic info
        self._width = int(self._vcap.get(CAP_PROP_FRAME_WIDTH))
        self._height = int(self._vcap.get(CAP_PROP_FRAME_HEIGHT))
        self._fps = int(round(self._vcap.get(CAP_PROP_FPS)))
        self._frame_cnt = int(self._vcap.get(CAP_PROP_FRAME_COUNT))
        self._fourcc = self._vcap.get(CAP_PROP_FOURCC)

    @property
    def vcap(self):
        """:obj:`cv2.VideoCapture`: raw VideoCapture object"""
        return self._vcap

    @property
    def opened(self):
        """bool: indicate whether the video is opened"""
        return self._vcap.isOpened()

    @property
    def width(self):
        """int: width of video frames"""
        return self._width

    @property
    def height(self):
        """int: height of video frames"""
        return self._height

    @property
    def resolution(self):
        """Tuple[int]: video resolution (width, height)"""
        return (self._width, self._height)

    @property
    def fps(self):
        """int: fps of the video"""
        return self._fps

    @property
    def frame_cnt(self):
        """int: total frames of the video"""
        return self._frame_cnt

    @property
    def fourcc(self):
        """str: "four character code" of the video"""
        return self._fourcc

    @property
    def position(self):
        """int: current cursor position, indicating which frame"""
        return self._position

    def _get_real_position(self):
        return int(round(self._vcap.get(CAP_PROP_POS_FRAMES)))

    def _set_real_position(self, frame_id):
        if self._position == frame_id == self._get_real_position():
            return
        self._vcap.set(CAP_PROP_POS_FRAMES, frame_id)
        pos = self._get_real_position()
        for _ in range(frame_id - pos):
            self._vcap.read()
        self._position = frame_id

    def read(self):
        """Read the next frame.

        If the next frame have been decoded before and in the cache, then
        return it directly, otherwise decode and return it, store in the cache.

        Returns:
            np.ndarray or None: return the frame if successful, otherwise None.
        """
        pos = self._position  # frame id to be read
        if self._cache:
            img = self._cache.get(pos)
            if img is not None:
                ret = True
            else:
                if self._position != self._get_real_position():
                    self._set_real_position(self._position)
                ret, img = self._vcap.read()
                if ret:
                    if self._to_rgb:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    self._cache.put(pos, img)
        else:
            ret, img = self._vcap.read()
        if ret:
            if self._to_rgb:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self._position = pos + 1
        return img

    def get_frame(self, frame_id):
        """Get frame by frame id.

        Args:
            frame_id (int): id of the expected frame, 0-based index.

        Returns:
            np.ndarray or None: return the frame if successful, otherwise None.
        """
        if frame_id < 0 or frame_id >= self._frame_cnt:
            raise ValueError(
                '"frame_id" must be [0, {}]'.format(self._frame_cnt - 1))
        if frame_id == self._position:
            return self.read()
        if self._cache:
            img = self._cache.get(frame_id)
            if img is not None:
                self._position = frame_id + 1
                return img
        self._set_real_position(frame_id)
        ret, img = self._vcap.read()
        if ret:
            if self._to_rgb:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self._position += 1
            if self._cache:
                self._cache.put(frame_id, img)
        return img

    def read_video(self):
        """Read the whole video as a list of images."""
        self._set_real_position(0)
        frames = [self.read() for _ in range(len(self))]
        return frames

    def cvt2frames(
        self,
        frame_dir,
        target_shape=None,
        file_start=0,
        filename_tmpl='{:06d}.jpg',
        start=0,
        max_num=0,
    ):
        """Convert a video to frame images.

        Args:
            frame_dir (str): output directory to store all the frame images.
            target_shape (Tuple[int], optional): resize and save in this shape.
                Default: None.
            file_start (int, optional): from which filename starts.
                Default: 0.
            filename_tmpl (str, optional): filename template, with the index
                as the variable. Default: '{:06d}.jpg'.
            start (int, optional): starting frame index.
                Default: 0.
            max_num (int, optional): maximum number of frames to be written.
                Default: 0.
        """
        mkdir_or_exist(frame_dir)
        if max_num <= 0:
            task_num = self.frame_cnt - start
        else:
            task_num = min(self.frame_cnt - start, max_num)
        if task_num <= 0:
            raise ValueError('start must be less than total frame number')
        if start >= 0:
            self._set_real_position(start)

        for i in range(task_num):
            img = self.read()
            if img is None:
                break
            filename = os.path.join(frame_dir,
                                 filename_tmpl.format(i + file_start))
            if target_shape is not None:
                img = resize(img, target_shape)
            cv2.imwrite(filename, img)

    def __len__(self):
        return self.frame_cnt

    def __getitem__(self, i):
        return self.get_frame(i)

    def __iter__(self):
        self._set_real_position(0)
        return self

    def __next__(self):
        img = self.read()
        if img is not None:
            return img
        else:
            raise StopIteration

    next = __next__

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._vcap.release()

def compact(l):
    return list(filter(None, l))


class BaseTransforms(object):
    """Data pre-processing steps."""

    def __init__(self, resolution, mean=(0.5, ), std=(0.5, )):
        self.transforms = transforms.Compose([
            transforms.ToTensor(),  # [3, H, W]
            transforms.Normalize(mean, std),  # [-1, 1]
            transforms.Resize(resolution),
        ])
        self.resolution = resolution

    def process_mask(self, mask):
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).long()
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)
            mask = F.resize(
                mask,
                self.resolution,
                interpolation=transforms.InterpolationMode.NEAREST)[0]
        else:
            mask = F.resize(
                mask,
                self.resolution,
                interpolation=transforms.InterpolationMode.NEAREST)
        return mask

    def __call__(self, input):
        return self.transforms(input)


def anno2mask(anno):
    """`anno` corresponds to `anno['frames'][i]` of a CLEVRER annotation."""
    masks = []
    for obj in anno['objects']:
        mask = mask_utils.decode(obj['mask'])
        masks.append(mask)
    masks = np.stack(masks, axis=0).astype(np.int32)  # [N, H, W]
    # put background mask at the first
    bg_mask = np.logical_not(np.any(masks, axis=0))[None]
    masks = np.concatenate([bg_mask, masks], axis=0)  # [N+1, H, W]
    return masks


def masks_to_boxes_pad(masks, num):
    """Extract bbox from mask, then pad to `num`.

    Args:
        masks: [N, H, W], masks of foreground objects
        num: int

    Returns:
        bboxes: [num, 4]
        pres_mask: [num], True means object, False means padded
    """
    masks = masks.clone()
    masks = masks[masks.sum([-1, -2]) > 0]
    bboxes = masks_to_boxes(masks).float()  # [N, 4]
    pad_bboxes = torch.zeros((num, 4)).type_as(bboxes)  # [num, 4]
    pad_bboxes[:bboxes.shape[0]] = bboxes
    pres_mask = torch.zeros(num).bool()  # [num]
    pres_mask[:bboxes.shape[0]] = True
    return pad_bboxes, pres_mask

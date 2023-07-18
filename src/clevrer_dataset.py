import os, random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision.transforms as transforms
from typing import Dict, List, Optional, Tuple

from clevrer_utils import load_obj, strip_suffix, read_img, VideoReader, compact, anno2mask, masks_to_boxes_pad

Batch = Dict[str, torch.Tensor]

class CLEVRERDataset(Dataset):
    """Dataset for loading CLEVRER videos."""

    def __init__(
        self,
        data_root='./data/CLEVRER',
        split='train',
        max_n_objects=6,
        video_len=128,
        n_sample_frames=6,
        warmup_len=5,
        frame_offset=None,
        load_mask=False,
        filter_enter=False,
    ):
        super().__init__()
        self.data_root = data_root
        self.split = split
        assert self.split in ['train', 'val', 'test']
        self.video_path = os.path.join(data_root, 'videos', split)
        self.anno_path = os.path.join(data_root, 'annotations', split)

        self.max_n_objects = max_n_objects
        self.video_len = video_len
        self.n_sample_frames = n_sample_frames
        self.warmup_len = warmup_len
        self.frame_offset = video_len // n_sample_frames if \
            frame_offset is None else frame_offset
        self.load_mask = load_mask
        self.filter_enter = filter_enter

        # all video paths
        self.files = self._get_files()
        self.num_videos = len(self.files)
        if self.filter_enter:
            self.valid_idx = self._get_filtered_sample_idx()
        else:
            self.valid_idx = self._get_sample_idx()

        # by default, we load small video clips
        self.load_video = False

        self.resolution = 64
        self.name = split + '_dataset'

        self.transform = transforms.Compose([
            transforms.ToTensor(),  # [3, H, W]
            transforms.Resize((self.resolution, self.resolution)),
        ])

    def _process_mask(self, mask):
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

    def _rand_another(self, is_video=False):
        """Random get another sample when encountering loading error."""
        if is_video:
            another_idx = np.random.choice(self.num_videos)
            return self.get_video(another_idx)
        another_idx = np.random.choice(len(self))
        return self.__getitem__(another_idx)

    def _get_video_start_idx(self, idx):
        return self.valid_idx[idx]

    def _read_frames(self, idx):
        """Read video frames. Directly read from jpg images if possible."""
        video_idx, start_idx = self._get_video_start_idx(idx)
        video_path = self.files[video_idx]
        frame_dir = strip_suffix(video_path)
        # if videos are not converted to frames, read from mp4 file
        if not os.path.isdir(frame_dir):
            cap = VideoReader(video_path)
            frames = [
                cap.get_frame(start_idx + n * self.frame_offset)
                for n in range(self.n_sample_frames)
            ]
        # otherwise, read from saved video frames
        else:
            # wrong video length
            if len(os.listdir(frame_dir)) != self.video_len:
                raise ValueError
            # read from jpg images
            frames = [
                read_img(
                    os.path.join(
                        frame_dir,
                        f'{start_idx + n * self.frame_offset:06d}.jpg'))
                for n in range(self.n_sample_frames)
            ]
        if any(frame is None for frame in frames):
            raise ValueError
        frames = [
            self.transform(Image.fromarray(img).convert('RGB'))
            for img in frames
        ]  # [T, C, H, W]
        return torch.stack(frames, dim=0).float()

    def _read_masks(self, idx):
        """Read masks from `derender_proposals`."""
        video_idx, start_idx = self._get_video_start_idx(idx)
        video_fn = os.path.basename(strip_suffix(self.files[video_idx]))
        anno_path = os.path.join(self.data_root, 'derender_proposals',
                                 f'proposal_{video_fn[-5:]}.json')
        anno = load_obj(anno_path)
        masks = [
            anno2mask(anno['frames'][start_idx + n * self.frame_offset])
            for n in range(self.n_sample_frames)
        ]  # [1+num_obj, H, W] each
        masks = [self._process_mask(mask)
                 for mask in masks]  # [1+num_obj, H, W] each
        bboxes = [
            masks_to_boxes_pad(mask[1:], self.max_n_objects + 1)
            for mask in masks
        ]  # ([max_num_obj, 4], [max_num_obj]) each
        masks = [mask.argmax(0) for mask in masks]  # [H, W]
        masks = torch.stack(masks, dim=0).long()  # [T, H, W]
        pres_mask = torch.stack([box[1] for box in bboxes], dim=0).bool()
        bboxes = torch.stack([box[0] for box in bboxes], dim=0).float()
        # masks: [T, H, W]
        # pres_mask: [T, max_num_obj]
        # bboxes: [T, max_num_obj, 4]
        return masks, pres_mask, bboxes

    def __getitem__(self, idx):
        """Data dict:
            - data_idx: int
            - observations: [T, 3, H, W]
            - error_flag: whether loading `idx` causes error and _rand_another
            - mask: [T, H, W], 0 is background
            - pres_mask: [T, max_num_obj], valid index of objects
            - bbox: [T, max_num_obj, 4], object bbox padded to `max_num_obj`
        """
        if self.load_video:
            return self.get_video(idx)

        try:
            frames = self._read_frames(idx)
            data_dict = {
                'data_idx': idx,
                'observations': frames,
                'error_flag': False,
            }
            if self.load_mask:
                masks, pres_masks, bboxes = self._read_masks(idx)
                data_dict['mask'] = masks
                data_dict['pres_mask'] = pres_masks
                data_dict['bbox'] = bboxes
        # corrupted video
        except ValueError:
            data_dict = self._rand_another()
            data_dict['error_flag'] = True
            return data_dict
        return data_dict

    def __len__(self):
        if self.load_video:
            return len(self.files)
        return len(self.valid_idx)

    def get_video(self, video_idx):
        video_path = self.files[video_idx]
        cap = VideoReader(video_path)  # , to_rgb=True)
        video = cap.read_video()
        # corrupted video
        if len(video) != self.video_len or any(img is None for img in video):
            data_dict = self._rand_another(is_video=True)
            data_dict['error_flag'] = True
            return data_dict
        video = [
            self.transform(Image.fromarray(frame).convert('RGB'))
            for frame in video[::self.frame_offset]
        ] # [T, C, H, W]
        return {
            'observations': torch.stack(video, dim=0),
            'error_flag': False,
            'data_idx': video_idx,
        }

    def _get_files(self):
        """Get path for all videos."""
        # test set doesn't have annotation json files
        if self.split == 'train':
            # start, end = 0, 10000
            start, end = 0, 1000
        elif self.split == 'val':
            start, end = 10000, 15000
        else:
            start, end = 15000, 20000
        video_paths = []
        for i in range(start, end):
            # annotation for this video is broken
            if i == 10800:
                continue
            level = i // 1000
            video_dir = f'video_{level * 1000:05d}-{(level + 1) * 1000:05d}'
            video_paths.append(
                os.path.join(self.video_path, video_dir, f'video_{i:05d}.mp4'))
        return sorted(compact(video_paths))

    def _get_sample_idx(self):
        """Get (video_idx, start_frame_idx) pairs as a list."""
        valid_idx = []
        for video_idx in range(len(self.files)):
            # simply use random uniform sampling
            max_start_idx = self.video_len - \
                (self.n_sample_frames - 1) * self.frame_offset
            if self.split == 'train':
                valid_idx += [(video_idx, idx) for idx in range(max_start_idx)]
            # in val/test we only sample each frame once
            else:
                size = self.n_sample_frames * self.frame_offset
                start_idx = []
                for idx in range(0, self.video_len - size + 1, size):
                    start_idx += [i + idx for i in range(self.frame_offset)]
                valid_idx += [(video_idx, idx) for idx in start_idx]
        return valid_idx

    def _get_enter_time(self, video_file):
        """Get the timestep when a new object enters the scene."""
        anno_file = video_file.\
            replace('video', 'annotation').replace('.mp4', '.json')
        anno = load_obj(anno_file)
        trajs = anno['motion_trajectory']
        num_objs = len(trajs[0]['objects'])
        all_t = []
        for i in range(len(trajs) - 1):
            cur_obj = trajs[i]['objects']
            next_obj = trajs[i + 1]['objects']
            for j in range(num_objs):
                if (not cur_obj[j]['inside_camera_view']) and \
                        next_obj[j]['inside_camera_view']:
                    all_t.append(i + 1)
                    break
        return all_t

    def _has_obj_enter(self, obj_enter_t, idx):
        """If any new object enters during prediction period."""
        MIN_FRAMES = 3
        t1 = idx + (self.warmup_len - 1 - MIN_FRAMES + 1) * self.frame_offset
        t2 = idx + (self.n_sample_frames - 1) * self.frame_offset
        for t in obj_enter_t:
            if t1 < t <= t2:
                return True
        return False

    def _get_filtered_sample_idx(self):
        """Get (video_idx, start_frame_idx) pairs as a list."""
        # filter out sequences with new objects entering
        # if an object's occurrence is less than `warmup` steps from clip end
        # we also filter it out
        valid_idx = []
        for video_idx, video_file in enumerate(self.files):
            enter_t = self._get_enter_time(video_file)
            max_start_idx = self.video_len - \
                (self.n_sample_frames - 1) * self.frame_offset
            if self.split == 'train':
                for idx in range(max_start_idx):
                    if self._has_obj_enter(enter_t, idx):
                        continue
                    valid_idx.append((video_idx, idx))
            # in val/test we only sample each frame once
            else:
                size = (self.n_sample_frames - 1) * self.frame_offset
                interval = size // 2
                for idx in range(0, self.video_len - size, interval):
                    # try to find one valid sequence from every block
                    max_idx = min(idx + interval, self.video_len - size)
                    for sub_idx in range(idx, max_idx, 1):
                        if self._has_obj_enter(enter_t, sub_idx):
                            continue
                        valid_idx.append((video_idx, sub_idx))
                        break
        return valid_idx

    def _collate_episodes_segments(self, episodes_segments: List[Dict]) -> Batch:
        batch = {}
        batch['observations'] = torch.stack([sampled_episode['observations'] for sampled_episode in episodes_segments])
        batch['observations'] = batch['observations'].float()  # int8 to float and scale
        return batch

    def sample_batch(self, batch_num_samples: int, sequence_length: int, weights: Optional[Tuple[float]] = None, sample_from_start: bool = True) -> Batch:
        return self._collate_episodes_segments(self._sample_episodes_segments(batch_num_samples, sequence_length, sample_from_start))

    def _sample_episodes_segments(self, batch_num_samples: int, sequence_length: int, sample_from_start: bool) -> List[Dict]:
        sampled_episode_ids = np.random.choice(range(len(self.files)), size=batch_num_samples)

        def pad(x):
            pad_right = torch.nn.functional.pad(x, [0 for _ in range(2 * x.ndim - 1)] + [padding_length_right]) if padding_length_right > 0 else x
            return torch.nn.functional.pad(pad_right, [0 for _ in range(2 * x.ndim - 2)] + [padding_length_left, 0]) if padding_length_left > 0 else pad_right

        sampled_episodes_segments = []
        for episode_idx in sampled_episode_ids:
            sampled_episode = self.__getitem__(episode_idx)
            t = sampled_episode['observations'].shape[0]
            
            if sample_from_start:
                start = random.randint(0, t - 1)
                stop = start + sequence_length
            else:
                stop = random.randint(1, t)
                start = stop - sequence_length

            padding_length_right = max(0, stop - t)
            padding_length_left = max(0, -start)

            # assert len(sampled_episodes_segments[-1]) == sequence_length
            sampled_episode['observations'] = sampled_episode['observations'][start:stop]
            if self.load_mask:
                sampled_episode['mask'] = sampled_episode['mask'][start:stop]
                sampled_episode['pres_mask'] = sampled_episode['pres_mask'][start:stop]
                sampled_episode['bbox'] = sampled_episode['bbox'][start:stop]

            sampled_episode['observations'] = pad(sampled_episode['observations'])
            if self.load_mask:
                sampled_episode['mask'] = pad(sampled_episode['mask'])
                sampled_episode['pres_mask'] = pad(sampled_episode['pres_mask'])
                sampled_episode['bbox'] = pad(sampled_episode['bbox'])

            sampled_episodes_segments.append(sampled_episode)
        return sampled_episodes_segments


    def traverse(self, batch_num_samples: int, chunk_size: int):
        pass

def build_clevrer_dataset(params, val_only=False, test_set=False):
    """Build CLEVRER video dataset."""
    args = dict(
        data_root=params.data_root,
        split='val',
        max_n_objects=6,
        n_sample_frames=params.n_sample_frames,
        warmup_len=params.input_frames,
        frame_offset=params.frame_offset,
        load_mask=params.get('load_mask', False),
        filter_enter=params.filter_enter,
    )

    if test_set:
        assert not val_only
        args['split'] = 'test'
        return CLEVRERDataset(**args)

    val_dataset = CLEVRERDataset(**args)
    if val_only:
        return val_dataset
    args['split'] = 'train'
    train_dataset = CLEVRERDataset(**args)
    return train_dataset, val_dataset

import cv2
import os
import json
import glob
from tqdm import tqdm
from torch import Tensor, squeeze
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from collections import OrderedDict
from abc import ABC, abstractmethod
from facenet_pytorch.models.mtcnn import MTCNN
from typing import List, Type, Optional
from loguru import logger


class VideoDataset(Dataset):
    """视频数据集
    类接受一个包含视频文件地址的列表作为输入，返回每一个视频文件的帧数据。
    具体来说，返回的元组结构为：`(视频地址，帧编号，帧信息(np.ndarray))`

    Args:
        videos (List[str]): 数据集视频路径列表
    """
    def __init__(self, videos: List[str], frame_interval: int = 5) -> None:
        super().__init__()
        self.videos_path = videos
        self.interval = frame_interval

    def __getitem__(self, index: int):
        video = self.videos_path[index]
        # 读取对应视频的所有帧
        capture = cv2.VideoCapture(video)
        frames_num = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        # 读取视频所有视频帧
        frames = OrderedDict()
        for i in range(frames_num):
            # 每隔interval读取一帧
            if i % self.interval != 0:
                capture.grab()
                continue
            capture.grab()
            success, frame = capture.retrieve()
            if not success:
                continue
            # 在OpenCV中，图像的默认颜色空间是BGR，而在其他大多数应用程序中，图像的默认颜色空间是RGB
            # 因此，在使用OpenCV处理图像时，需要将BGR格式的图像转换为RGB格式的图像
            # 这里返回的是 (height, weight, channel)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # frame = Image.fromarray(frame)
            # frame = frame.resize(size=[s // 2 for s in frame.size])
            frames[i] = frame
        return video, list(frames.keys()), list(frames.values())

    def __len__(self) -> int:
        return len(self.videos_path)


class VideoFaceDetector(ABC):

    def __init__(self, **kwargs) -> None:
        super().__init__()

    @property
    @abstractmethod
    def _batch_size(self) -> int:
        pass

    @abstractmethod
    def _detect_faces(self, frames) -> List:
        pass


class FacenetDetector(VideoFaceDetector):
    """人脸检测器
    使用MTCNN作为人脸检测的检测器，可调用 `_detect_faces` 方法进行人脸检测，该方法的返回结果如下所示
    1. 当传入的frames位列表时，返回一个三维列表。一维表示各个图片，第二维表示每个图片中的各个人脸，第三维表示人脸框的左上和右下角坐标
    ```
    [[
        [66.39105224609375, 72.21067810058594, 157.32815551757812, 195.536376953125],
        [311.5481262207031, 74.7406997680664, 386.9224853515625, 173.45916748046875]
    ]]
    ```
    2. 当传入的frames位一个图片时，返回一个二维列表，第一维表示图片中的各个人脸，第二维表示人脸框的左上和右下角坐标
    ```
    [
        [66.39105224609375, 72.21067810058594, 157.32815551757812, 195.536376953125],
        [311.5481262207031, 74.7406997680664, 386.9224853515625, 173.45916748046875]
    ]
    3. 没有检测到人脸时返回None
    ```
    """    
    def __init__(self, device: str) -> None:
        super().__init__()
        self.detector = MTCNN(margin=0, thresholds=[0.85, 0.95, 0.95], device=device)

    def _detect_faces(self, frames) -> Optional[List]:
        batch_boxes, *_ = self.detector.detect(frames, landmarks=False)
        # 没有探测到人脸的图像，返回None
        if batch_boxes is None:
            return None
        return [b.tolist() if b is not None else None for b in batch_boxes]

    @property
    def _batch_size(self):
        return 32
    

def get_all_videopath(folder_path: str) -> List[str]:
    """获取文件夹下所有的视频文件路径

    Args:
        folder_path (str): 文件夹路径

    Returns:
        List[str]: 视频文件路径
    """
    video_paths = []
    for ext in ["*.mp4", "*.avi", "*.mov", "*.flv", "*.wmv"]:
        video_paths.extend(glob.glob(os.path.join(folder_path, ext)))
    return video_paths


def detect_faces_in_videos(dataset_path: str, detector_cls: Type[VideoFaceDetector] = FacenetDetector, device: str = 'cpu', out_dir: Optional[str] = None, load_worker: int = 2, frame_interval: int = 5):
    """检测批量视频中的人脸，并将结果以json形式保存至文件

    Args:
        dataset_path (str): 批量视频的存储路径
        detector_cls (Type[VideoFaceDetector], optional): 所使用的检测器. Defaults to FacenetDetector.
        device (str, optional): 在哪个硬件设备上运行检测. Defaults to 'cpu'.
        out_dir (Optional[str], optional): json文件输出的位置. Defaults to None.
        load_worker (int, optional): 加载视频时使用几个线程. Defaults to 2.
        frame_interval (int, optional): 读取帧时的间隔. Defaults to 5.
    """    
    detector = detector_cls(device=device)
    dataset = VideoDataset(get_all_videopath(dataset_path), frame_interval)
    loader = DataLoader(dataset, shuffle = False, num_workers = load_worker, batch_size = 1)
    missed_video_paths = []
    
    # 每次处理一个视频的frame
    for item in tqdm(loader):
        '''
        result这个字典的结构如下
        {
            帧编号: 一个二维列表，第一维表示图片中的各个人脸，第二维表示人脸框的左上和右下角坐标
            ...
        }
        '''
        movie_result = {}
        # 这里的frames是List对象
        video_path, indices, frames = item
        # 视频文件名作为其id
        video_id = os.path.basename(video_path[0]).split('.')[0]
        logger.info('Start detect faces in video {}'.format(video_id))
        # 丢掉frame没用的第一维，使其变为三维 (h, w, c)
        for i in range(len(frames)):
            frames[i] = squeeze(frames[i], dim = 0)
        
        # TODO 并行化改进
        # MTCNN探测每一个帧的人脸
        for idx, frame_idx in enumerate(indices):
            frame_idx = frame_idx[0].item()
            frame_detect_res = detector._detect_faces(frames[idx])
            if frame_detect_res is not None:
                movie_result[frame_idx] = frame_detect_res
                
        # 保存结果至json文件中
        if len(movie_result) > 0:
            logger.info('Detect {} frames have face in video {}'.format(str(len(movie_result)), video_id))
            if out_dir is not None:
                os.makedirs(out_dir, exist_ok=True)
                with open(os.path.join(out_dir, "{}.json".format(video_id)), "w", encoding='utf-8') as f:
                    json.dump(movie_result, f, indent=4, sort_keys=True)
                    logger.info('Video {} frame info save to {} json file'.format(video_id, f.name))
        else:
            missed_video_paths.append(video_path)
        
    if len(missed_video_paths) > 0:
        logger.warning("The detector did not find faces inside the following videos:")
        logger.warning(missed_video_paths)
        logger.warning("We suggest to re-run the code decreasing the detector threshold.")


def crop_faces_single_video(video_path: str, frame_json_path: str, out_root: str) -> bool:
    """根据识别出来的人脸信息截取单个视频中帧的人脸图片，并将人脸图片保存。
    截取的人脸图像保存在 `out_root/id` 的文件夹中，其中 `id` 为视频文件的文件名（无扩展名的文件名）

    Args:
        video_path (str): 视频地址
        frame_json_path (str): 每一帧人脸识别的结果保存的json文件地址
        out_root (str): 人脸图片保存地址

    Returns:
        bool: 是否成功截取人脸
    """    
    try:        
        # 读取方框信息
        with open(frame_json_path, "r", encoding='utf-8') as f:
            video_boxes_dict = json.load(f)
        
        # 创建视频输出目录
        id = os.path.basename(video_path).split('.')[0]
        out_dir = os.path.join(out_root, id)
        os.makedirs(out_dir, exist_ok=True)
        logger.info('Create frame output folder at {}'.format(out_dir))
        
        # 打开视频
        capture = cv2.VideoCapture(video_path)
        frames_num = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        counter = 0
        # 处理每一帧的框
        for frame_idx in range(frames_num):
            capture.grab()
            success, frame = capture.retrieve()
            # 获取对应帧的人脸框信息
            frame_boxes = video_boxes_dict.get(str(frame_idx))
            if not success or frame_boxes is None:
                continue
            counter += 1
            
            # 将每个边界框（bounding box）中的目标物体从视频帧中裁剪出来
            crops = []
            for bbox in frame_boxes:
                xmin, ymin, xmax, ymax = [b for b in bbox]
                w = xmax - xmin
                h = ymax - ymin
                # p_w和p_h是用来计算裁剪后图像的填充量的变量
                # 如果目标物体的高度大于宽度，则在图像的左右两侧添加填充，以使其宽度与高度相等; 如果目标物体的宽度大于高度，则在图像的上下两侧添加填充，以使其高度与宽度相等
                # 这样做是为了确保裁剪后的图像具有相同的高度和宽度，以便于后续处理
                p_h = 0
                p_w = 0
                if h > w:
                    p_w = int((h - w) / 2)
                elif h < w:
                    p_h = int((w - h) / 2)
                # 从视频帧中裁剪出一个矩形区域，该区域的左上角坐标为(max(xmin - p_w, 0), max(ymin - p_h, 0))，右下角坐标为(xmax + p_w, ymax + p_h)
                crop = frame[int(max(ymin - p_h, 0)) : int(ymax + p_h), int(max(xmin - p_w, 0)) : int(xmax + p_w)]
                h, w = crop.shape[:2]
                crops.append(crop)
            # 保存裁剪的物体图片
            for frame_box_idx, crop in enumerate(crops):
                cv2.imwrite(os.path.join(out_dir, "{}_{}.png".format(frame_idx, frame_box_idx)), crop)
                logger.info('Crop picture from video {} at frame {}'.format(id, frame_idx))
        # 如果没有人脸的话就输出提醒一下
        if counter == 0:
            logger.warning('No face detected in video {}, path is {}'.format(id, video_path))
            return False
        return True
    except Exception as e:
        logger.exception('Exception occured in crop face image from video {}'.format(video_path))
        logger.exception('Exception info:')
        logger.exception(e)
        return False


def crop_faces_videos(dataset_path: str, frames_json_path: str, out_root: str):
    """根绝识别出的人脸数量截取批量视频中的人脸

    Args:
        dataset_path (str): 批量视频存储地址
        frames_json_path (str): 存储视频人脸识别结果的地址
        out_root (str): 存储人脸图片的路径
    """    
    videos_path = get_all_videopath(dataset_path)
    for video_path in tqdm(videos_path):
        id = os.path.basename(video_path).split('.')[0]
        logger.info('Start crop faces from video {}'.format(id))
        
        frame_json_path = os.path.join(frames_json_path, id + '.json')
        if not os.path.exists(frame_json_path):
            logger.warning('Detect result (json file) of video {} not found, video path is {}'.format(id, video_path))
            continue
        res = crop_faces_single_video(video_path, frame_json_path, out_root)
        if res:
            logger.info('Video {} crop finished, path is {}'.format(id, video_path))
        else:
            logger.warning('Video {} crop failed, path is {}'.format(id, video_path))

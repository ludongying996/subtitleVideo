import subprocess
import os
import json
import time
import tempfile
import shutil
import sys
import glob

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("[VideoSubtitleNode] 警告: requests库未安装，VideoSubtitleNode(URL)节点将不可用")

FFMPEG_PATH = None
FFPROBE_PATH = None

def find_ffmpeg():
    global FFMPEG_PATH, FFPROBE_PATH
    
    if FFMPEG_PATH is not None:
        return True
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    is_windows = sys.platform.startswith('win')
    
    if is_windows:
        local_ffmpeg_paths = [
            os.path.join(current_dir, "ffmpeg-master-latest-win64-gpl-shared", "bin", "ffmpeg.exe"),
            os.path.join(current_dir, "ffmpeg", "bin", "ffmpeg.exe"),
            os.path.join(current_dir, "ffmpeg", "ffmpeg.exe"),
            os.path.join(current_dir, "bin", "ffmpeg.exe"),
            os.path.join(current_dir, "ffmpeg.exe"),
        ]
        
        local_ffprobe_paths = [
            os.path.join(current_dir, "ffmpeg-master-latest-win64-gpl-shared", "bin", "ffprobe.exe"),
            os.path.join(current_dir, "ffmpeg", "bin", "ffprobe.exe"),
            os.path.join(current_dir, "ffmpeg", "ffprobe.exe"),
            os.path.join(current_dir, "bin", "ffprobe.exe"),
            os.path.join(current_dir, "ffprobe.exe"),
        ]
        
        search_names_ffmpeg = ['ffmpeg', 'ffmpeg.exe']
        search_names_ffprobe = ['ffprobe', 'ffprobe.exe']
    else:
        local_ffmpeg_paths = [
            os.path.join(current_dir, "ffmpeg-master-latest-linux64-gpl", "bin", "ffmpeg"),
            os.path.join(current_dir, "ffmpeg-master-latest-linux64-gpl", "ffmpeg"),
            os.path.join(current_dir, "ffmpeg", "bin", "ffmpeg"),
            os.path.join(current_dir, "ffmpeg", "ffmpeg"),
            os.path.join(current_dir, "bin", "ffmpeg"),
            os.path.join(current_dir, "ffmpeg"),
            "/usr/bin/ffmpeg",
            "/usr/local/bin/ffmpeg",
        ]
        
        local_ffprobe_paths = [
            os.path.join(current_dir, "ffmpeg-master-latest-linux64-gpl", "bin", "ffprobe"),
            os.path.join(current_dir, "ffmpeg-master-latest-linux64-gpl", "ffprobe"),
            os.path.join(current_dir, "ffmpeg", "bin", "ffprobe"),
            os.path.join(current_dir, "ffmpeg", "ffprobe"),
            os.path.join(current_dir, "bin", "ffprobe"),
            os.path.join(current_dir, "ffprobe"),
            "/usr/bin/ffprobe",
            "/usr/local/bin/ffprobe",
        ]
        
        search_names_ffmpeg = ['ffmpeg']
        search_names_ffprobe = ['ffprobe']
    
    for path in local_ffmpeg_paths:
        if os.path.isfile(path):
            FFMPEG_PATH = path
            print(f"[VideoSubtitleNode] 找到本地FFmpeg: {path}")
            break
    
    for path in local_ffprobe_paths:
        if os.path.isfile(path):
            FFPROBE_PATH = path
            print(f"[VideoSubtitleNode] 找到本地FFprobe: {path}")
            break
    
    if FFMPEG_PATH is None:
        for name in search_names_ffmpeg:
            path = shutil.which(name)
            if path:
                FFMPEG_PATH = path
                print(f"[VideoSubtitleNode] 从PATH找到FFmpeg: {path}")
                break
    
    if FFPROBE_PATH is None:
        for name in search_names_ffprobe:
            path = shutil.which(name)
            if path:
                FFPROBE_PATH = path
                print(f"[VideoSubtitleNode] 从PATH找到FFprobe: {path}")
                break
    
    if FFMPEG_PATH is None:
        print("[VideoSubtitleNode] 警告: 未找到FFmpeg")
        FFMPEG_PATH = 'ffmpeg'
    
    if FFPROBE_PATH is None:
        FFPROBE_PATH = 'ffprobe'
    
    return True

find_ffmpeg()

CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "subtitle_config.json")

if os.path.exists(CONFIG_FILE):
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception as e:
        print(f"[VideoSubtitleNode] 配置文件读取失败: {e}")
        config = None

if not config:
    config = {
        "font": {"name": "Microsoft YaHei", "size": 13},
        "style": {
            "primary_color": "&Hffffff",
            "outline_color": "&H000000",
            "outline": 1,
            "shadow": 1,
            "alignment": 2,
            "margin_v": 40
        },
        "merge": {"min_duration": 0.5},
        "max_chars": {"vertical": 12, "horizontal": 20}
    }
    print("[VideoSubtitleNode] 使用默认配置")

font_name = config["font"]["name"]
font_size = config["font"]["size"]
style = config["style"]
merge_config = config["merge"]
max_chars_config = config["max_chars"]

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "comfyui_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def tensor_to_video(tensor, output_path, fps=30, use_gpu=False):
    if not NUMPY_AVAILABLE or not PIL_AVAILABLE:
        raise ImportError("numpy和PIL库未安装")
    
    print(f"[VideoSubtitleFromImages] 保存视频: {output_path}")
    
    num_frames = tensor.shape[0]
    height = tensor.shape[1]
    width = tensor.shape[2]
    
    temp_dir = tempfile.mkdtemp(prefix="subtitle_frames_")
    
    try:
        print(f"[VideoSubtitleFromImages] 保存临时帧文件...")
        
        for i in range(num_frames):
            frame = tensor[i].cpu().numpy()
            frame_uint8 = (frame * 255).astype(np.uint8)
            img = Image.fromarray(frame_uint8)
            frame_path = os.path.join(temp_dir, f"frame_{i:06d}.png")
            img.save(frame_path)
            
            if (i + 1) % 100 == 0:
                print(f"[VideoSubtitleFromImages] 已保存 {i + 1}/{num_frames} 帧")
        
        frame_pattern = os.path.join(temp_dir, "frame_%06d.png")
        
        if use_gpu and check_nvidia_gpu():
            cmd = [
                FFMPEG_PATH,
                "-y",
                "-framerate", str(fps),
                "-i", frame_pattern,
                "-c:v", "h264_nvenc",
                "-preset", "p4",
                "-cq", "23",
                "-pix_fmt", "yuv420p",
                output_path
            ]
        else:
            cmd = [
                FFMPEG_PATH,
                "-y",
                "-framerate", str(fps),
                "-i", frame_pattern,
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                output_path
            ]
        
        print(f"[VideoSubtitleFromImages] 编码视频...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"[VideoSubtitleFromImages] 视频编码失败: {result.stderr}")
            return None
        
        if os.path.exists(output_path):
            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"[VideoSubtitleFromImages] 视频保存成功: {output_path}")
            print(f"[VideoSubtitleFromImages] 文件大小: {file_size_mb:.2f} MB")
            return output_path
        
        return None
        
    finally:
        for f in os.listdir(temp_dir):
            try:
                os.remove(os.path.join(temp_dir, f))
            except:
                pass
        try:
            os.rmdir(temp_dir)
        except:
            pass


def check_nvidia_gpu():
    if TORCH_AVAILABLE and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[VideoSubtitleNode] 检测到GPU: {gpu_name}")
        return True
    return False


def get_video_info(video_path):
    cmd = [
        FFPROBE_PATH,
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    data = json.loads(result.stdout)
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            width = int(stream.get("width", 0))
            height = int(stream.get("height", 0))
            return width, height
    return 0, 0


def is_vertical_video(width, height):
    return height > width


def clean_text(text):
    text = text.replace(" ", "")
    text = text.replace("　", "")
    while text and text[-1] in "，。、,":
        text = text[:-1]
    return text


def parse_subtitle_line(line):
    time_part, text = line.split(")", 1)
    start, end = time_part.replace("(", "").split(",")
    return float(start), float(end), text.strip()


def merge_short_subtitles(lines, min_duration=None, max_len=None):
    if min_duration is None:
        min_duration = merge_config["min_duration"]
    
    parsed = []
    for line in lines:
        line = line.strip()
        if not line: continue
        try:
            start, end, text = parse_subtitle_line(line)
            text = clean_text(text)
            parsed.append((start, end, text))
        except:
            continue
    
    merged = []
    i = 0
    while i < len(parsed):
        start, end, text = parsed[i]
        duration = end - start
        
        j = i + 1
        while j < len(parsed) and duration < min_duration:
            next_start, next_end, next_text = parsed[j]
            combined = text + next_text
            if len(combined) <= max_len:
                text = combined
                end = next_end
                duration = end - start
                j += 1
            else:
                break
        
        merged.append((start, end, text))
        i = j
    
    return merged


def wrap_text(text, max_chars):
    text = text.strip()
    if len(text) <= max_chars:
        return text
    
    lines = []
    current_line = ""
    
    for char in text:
        if char == '\n':
            if current_line:
                lines.append(current_line)
                current_line = ""
        elif len(current_line) >= max_chars:
            lines.append(current_line)
            current_line = char
        else:
            current_line += char
    
    if current_line:
        lines.append(current_line)
    
    return "\n".join(lines)


def time_format(s):
    s = float(s)
    hours = int(s // 3600)
    mins = int((s % 3600) // 60)
    secs = s % 60
    return f"{hours:02}:{mins:02}:{secs:06.3f}"


def download_video(url, output_path):
    if not REQUESTS_AVAILABLE:
        raise ImportError("requests库未安装，无法下载视频")
    
    print(f"[VideoSubtitleNode] 下载视频: {url}")
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()
    
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    print(f"[VideoSubtitleNode] 下载完成: {output_path}")
    return output_path


def video_to_frames(video_path, output_dir):
    print(f"[VideoSubtitleNode] 提取视频帧: {video_path}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    frame_pattern = os.path.join(output_dir, "frame_%06d.png")
    
    cmd = [
        FFMPEG_PATH,
        "-i", video_path,
        "-vsync", "0",
        "-q:v", "2",
        "-y",
        frame_pattern
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"[VideoSubtitleNode] 提取帧失败: {result.stderr}")
        return []
    
    frames = sorted(glob.glob(os.path.join(output_dir, "frame_*.png")))
    print(f"[VideoSubtitleNode] 提取了 {len(frames)} 帧")
    
    return frames


def frames_to_tensor(frame_paths):
    if not NUMPY_AVAILABLE or not PIL_AVAILABLE:
        raise ImportError("numpy和PIL库未安装")
    
    if not frame_paths:
        return None
    
    print(f"[VideoSubtitleNode] 转换 {len(frame_paths)} 帧为tensor...")
    
    frames = []
    for path in frame_paths:
        img = Image.open(path).convert("RGB")
        img_array = np.array(img).astype(np.float32) / 255.0
        frames.append(img_array)
    
    tensor = torch.from_numpy(np.stack(frames, axis=0))
    
    print(f"[VideoSubtitleNode] tensor形状: {tensor.shape}")
    
    return tensor


try:
    from PIL import ImageDraw, ImageFont
    PIL_DRAW_AVAILABLE = True
except ImportError:
    PIL_DRAW_AVAILABLE = False


def find_chinese_font(font_name="微软雅黑"):
    is_windows = sys.platform.startswith('win')
    
    if is_windows:
        font_mappings = {
            "微软雅黑": ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/msyhbd.ttc"],
            "黑体": ["C:/Windows/Fonts/simhei.ttf"],
            "宋体": ["C:/Windows/Fonts/simsun.ttc"],
            "楷体": ["C:/Windows/Fonts/simkai.ttf"],
        }
        
        default_fonts = [
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/simhei.ttf",
            "C:/Windows/Fonts/simsun.ttc",
            "C:/Windows/Fonts/arial.ttf",
        ]
    else:
        font_mappings = {
            "微软雅黑": [],
            "黑体": ["/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"],
            "宋体": ["/usr/share/fonts/truetype/arphic/uming.ttc"],
        }
        
        default_fonts = [
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        ]
    
    if font_name in font_mappings:
        for path in font_mappings[font_name]:
            if os.path.exists(path):
                print(f"[VideoSubtitleNode] 找到字体文件: {path}")
                return path
    
    for path in default_fonts:
        if os.path.exists(path):
            print(f"[VideoSubtitleNode] 使用默认字体: {path}")
            return path
    
    print(f"[VideoSubtitleNode] 警告: 未找到中文字体，使用系统默认")
    return None


def draw_subtitle_on_frame(img_array, text, font_size, font_name, style_config, max_chars):
    if not PIL_DRAW_AVAILABLE:
        return img_array
    
    img = Image.fromarray((img_array * 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    
    font_path = find_chinese_font(font_name)
    
    if font_path and os.path.exists(font_path):
        try:
            font = ImageFont.truetype(font_path, font_size)
        except Exception as e:
            print(f"[VideoSubtitleNode] 字体加载失败: {e}")
            font = ImageFont.load_default()
    else:
        try:
            font = ImageFont.truetype(font_name, font_size)
        except:
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except:
                font = ImageFont.load_default()
    
    wrapped_text = wrap_text(text, max_chars)
    lines = wrapped_text.split("\n")
    
    primary_color = (255, 255, 255)
    outline_color = (0, 0, 0)
    outline_width = style_config.get("outline", 1)
    
    img_width, img_height = img.size
    margin_v = style_config.get("margin_v", 40)
    
    line_heights = []
    line_widths = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_heights.append(bbox[3] - bbox[1])
        line_widths.append(bbox[2] - bbox[0])
    
    total_height = sum(line_heights) + (len(lines) - 1) * 5
    
    y = img_height - total_height - margin_v
    
    for i, line in enumerate(lines):
        line_width = line_widths[i]
        x = (img_width - line_width) // 2
        
        for ox in range(-outline_width, outline_width + 1):
            for oy in range(-outline_width, outline_width + 1):
                if ox != 0 or oy != 0:
                    draw.text((x + ox, y + oy), line, font=font, fill=outline_color)
        
        draw.text((x, y), line, font=font, fill=primary_color)
        y += line_heights[i] + 5
    
    return np.array(img).astype(np.float32) / 255.0


def process_images_with_subtitle(images, subtitle_text, fps=30, 
                                  font_name_param=None, font_size_param=None,
                                  outline_param=None, shadow_param=None, margin_v_param=None,
                                  max_chars_vertical=None, max_chars_horizontal=None,
                                  use_gpu=False):
    if not NUMPY_AVAILABLE or not PIL_AVAILABLE:
        raise ImportError("numpy和PIL库未安装")
    
    print(f"[VideoSubtitleFromImages] 处理图像帧: {images.shape}")
    
    if font_name_param is None:
        font_name_param = font_name
    if font_size_param is None:
        font_size_param = font_size
    if outline_param is None:
        outline_param = style.get("outline", 1)
    if shadow_param is None:
        shadow_param = style.get("shadow", 1)
    if margin_v_param is None:
        margin_v_param = style.get("margin_v", 40)
    if max_chars_vertical is None:
        max_chars_vertical = max_chars_config.get("vertical", 12)
    if max_chars_horizontal is None:
        max_chars_horizontal = max_chars_config.get("horizontal", 20)
    
    style_param = {
        "outline": outline_param,
        "shadow": shadow_param,
        "margin_v": margin_v_param
    }
    
    num_frames = images.shape[0]
    height = images.shape[1]
    width = images.shape[2]
    
    vertical = height > width
    print(f"[VideoSubtitleFromImages] 分辨率: {width}x{height}, {'竖屏' if vertical else '横屏'}")
    
    current_font_size = font_size_param
    if not vertical:
        current_font_size = int(font_size_param * 1.5)
    
    if vertical:
        max_chars = max_chars_vertical
    else:
        max_chars = max_chars_horizontal
    
    lines = subtitle_text.strip().split("\n")
    merged_subtitles = merge_short_subtitles(lines, max_len=max_chars)
    
    print(f"[VideoSubtitleFromImages] 字幕条数: {len(merged_subtitles)}")
    
    if use_gpu and TORCH_AVAILABLE and torch.cuda.is_available():
        print(f"[VideoSubtitleFromImages] 使用GPU加速处理")
        images_gpu = images.cuda()
    else:
        images_gpu = images
    
    result_frames = []
    
    for frame_idx in range(num_frames):
        current_time = frame_idx / fps
        
        current_text = ""
        for start, end, text in merged_subtitles:
            if start <= current_time < end:
                current_text = text
                break
        
        frame = images_gpu[frame_idx].cpu().numpy()
        
        if current_text:
            frame = draw_subtitle_on_frame(
                frame, current_text, current_font_size, 
                font_name_param, style_param, max_chars
            )
        
        result_frames.append(frame)
        
        if (frame_idx + 1) % 100 == 0:
            print(f"[VideoSubtitleFromImages] 已处理 {frame_idx + 1}/{num_frames} 帧")
    
    result = torch.from_numpy(np.stack(result_frames, axis=0))
    print(f"[VideoSubtitleFromImages] 处理完成，输出形状: {result.shape}")
    
    return result


def process_subtitle(video_path, subtitle_text, output_path, use_gpu=True):
    width, height = get_video_info(video_path)
    vertical = is_vertical_video(width, height)
    
    print(f"[VideoSubtitleNode] 视频分辨率: {width}x{height}, {'竖屏' if vertical else '横屏'}")
    
    current_font_size = font_size
    if not vertical:
        current_font_size = int(font_size * 1.5)
    
    if vertical:
        max_chars = max_chars_config["vertical"]
    else:
        max_chars = max_chars_config["horizontal"]
    
    srt_file = output_path.replace('.mp4', '.srt')
    
    lines = subtitle_text.strip().split("\n")
    merged_subtitles = merge_short_subtitles(lines, max_len=max_chars)
    
    srt_content = ""
    index = 1
    for start, end, text in merged_subtitles:
        start_str = time_format(start)
        end_str = time_format(end)
        
        wrapped_text = wrap_text(text, max_chars)
        srt_content += f"{index}\n{start_str} --> {end_str}\n{wrapped_text}\n\n"
        index += 1
    
    with open(srt_file, "w", encoding="utf-8") as f:
        f.write(srt_content)
    
    style_params = (
        f"FontName={font_name},"
        f"FontSize={current_font_size},"
        f"PrimaryColour={style['primary_color']},"
        f"OutlineColour={style['outline_color']},"
        f"Outline={style['outline']},"
        f"Shadow={style['shadow']},"
        f"Alignment={style['alignment']},"
        f"MarginV={style['margin_v']}"
    )
    
    srt_path_escaped = srt_file.replace("\\", "/").replace(":", "\\:")
    
    if use_gpu and check_nvidia_gpu():
        print("[VideoSubtitleNode] 使用GPU加速 (NVENC)")
        ffmpeg_cmd = [
            FFMPEG_PATH,
            "-loglevel", "warning",
            "-i", video_path,
            "-vf", f"subtitles='{srt_path_escaped}':force_style='{style_params}'",
            "-c:v", "h264_nvenc",
            "-preset", "p4",
            "-cq", "23",
            "-c:a", "aac",
            "-b:a", "128k",
            "-y", output_path
        ]
    else:
        print("[VideoSubtitleNode] 使用CPU编码")
        ffmpeg_cmd = [
            FFMPEG_PATH,
            "-loglevel", "warning",
            "-i", video_path,
            "-vf", f"subtitles='{srt_path_escaped}':force_style='{style_params}'",
            "-c:v", "libx264",
            "-crf", "23",
            "-preset", "medium",
            "-c:a", "aac",
            "-b:a", "128k",
            "-y", output_path
        ]
    
    print(f"[VideoSubtitleNode] 执行FFmpeg命令...")
    result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"[VideoSubtitleNode] FFmpeg错误: {result.stderr}")
        return None
    
    if os.path.exists(srt_file):
        os.remove(srt_file)
    
    return output_path


class VideoSubtitleNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_url": ("STRING", {"default": "", "multiline": False}),
                "subtitle_text": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "use_gpu": ("BOOLEAN", {"default": True}),
                "字体名称": ("STRING", {"default": "微软雅黑"}),
                "字体大小": ("INT", {"default": 13, "min": 8, "max": 100}),
                "描边宽度": ("INT", {"default": 1, "min": 0, "max": 10}),
                "阴影宽度": ("INT", {"default": 1, "min": 0, "max": 10}),
                "底部边距": ("INT", {"default": 40, "min": 0, "max": 200}),
                "竖屏每行最大字符数": ("INT", {"default": 12, "min": 5, "max": 50}),
                "横屏每行最大字符数": ("INT", {"default": 20, "min": 5, "max": 100}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "process"
    CATEGORY = "video"
    
    def process(self, video_url, subtitle_text, use_gpu=True,
                字体名称="微软雅黑", 字体大小=13, 描边宽度=1, 阴影宽度=1, 底部边距=40,
                竖屏每行最大字符数=12, 横屏每行最大字符数=20):
        print(f"\n{'='*60}")
        print(f"[VideoSubtitleNode] 开始处理")
        print(f"  字体: {字体名称}, 大小: {字体大小}")
        print(f"  描边: {描边宽度}, 阴影: {阴影宽度}, 底部边距: {底部边距}")
        print(f"  竖屏最大字符: {竖屏每行最大字符数}, 横屏最大字符: {横屏每行最大字符数}")
        print(f"  GPU加速: {'开启' if use_gpu else '关闭'}")
        print(f"{'='*60}")
        
        if not video_url:
            print("[VideoSubtitleNode] 错误: 视频URL为空")
            return (torch.zeros(1, 64, 64, 3),)
        
        if not subtitle_text:
            print("[VideoSubtitleNode] 错误: 字幕文本为空")
            return (torch.zeros(1, 64, 64, 3),)
        
        start_time = time.time()
        
        global font_size, font_name, style, max_chars_config
        old_font_size = font_size
        old_font_name = font_name
        old_style = style.copy()
        old_max_chars = max_chars_config.copy()
        
        font_size = 字体大小
        font_name = 字体名称
        style["outline"] = 描边宽度
        style["shadow"] = 阴影宽度
        style["margin_v"] = 底部边距
        max_chars_config["vertical"] = 竖屏每行最大字符数
        max_chars_config["horizontal"] = 横屏每行最大字符数
        
        timestamp = int(time.time())
        temp_video = os.path.join(OUTPUT_DIR, f"input_{timestamp}.mp4")
        output_video = os.path.join(OUTPUT_DIR, f"output_{timestamp}.mp4")
        frames_dir = os.path.join(OUTPUT_DIR, f"frames_{timestamp}")
        
        try:
            download_video(video_url, temp_video)
            
            result = process_subtitle(temp_video, subtitle_text, output_video, use_gpu)
            
            font_size = old_font_size
            font_name = old_font_name
            style = old_style
            max_chars_config = old_max_chars
            
            if result and os.path.exists(result):
                file_size_mb = os.path.getsize(result) / (1024 * 1024)
                elapsed_time = time.time() - start_time
                
                print(f"\n[VideoSubtitleNode] 视频处理完成!")
                print(f"  输出文件: {result}")
                print(f"  文件大小: {file_size_mb:.2f} MB")
                print(f"  处理耗时: {elapsed_time:.2f} 秒")
                
                frames = video_to_frames(result, frames_dir)
                
                if frames:
                    tensor = frames_to_tensor(frames)
                    
                    for f in frames:
                        if os.path.exists(f):
                            os.remove(f)
                    if os.path.exists(frames_dir):
                        os.rmdir(frames_dir)
                    
                    if os.path.exists(temp_video):
                        os.remove(temp_video)
                    if os.path.exists(output_video):
                        os.remove(output_video)
                    
                    print(f"[VideoSubtitleNode] 返回图像tensor: {tensor.shape}")
                    return (tensor,)
                
            print("[VideoSubtitleNode] 处理失败")
            return (torch.zeros(1, 64, 64, 3),)
                
        except Exception as e:
            font_size = old_font_size
            font_name = old_font_name
            style = old_style
            max_chars_config = old_max_chars
            
            print(f"[VideoSubtitleNode] 处理异常: {str(e)}")
            if os.path.exists(temp_video):
                os.remove(temp_video)
            return (torch.zeros(1, 64, 64, 3),)


class VideoSubtitleFromImages:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "subtitle_text": ("STRING", {"default": "", "multiline": True}),
                "fps": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 120.0}),
            },
            "optional": {
                "use_gpu": ("BOOLEAN", {"default": False}),
                "字体名称": ("STRING", {"default": "微软雅黑"}),
                "字体大小": ("INT", {"default": 13, "min": 8, "max": 100}),
                "描边宽度": ("INT", {"default": 1, "min": 0, "max": 10}),
                "阴影宽度": ("INT", {"default": 1, "min": 0, "max": 10}),
                "底部边距": ("INT", {"default": 40, "min": 0, "max": 200}),
                "竖屏每行最大字符数": ("INT", {"default": 12, "min": 5, "max": 50}),
                "横屏每行最大字符数": ("INT", {"default": 20, "min": 5, "max": 100}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "process"
    CATEGORY = "video"
    
    def process(self, images, subtitle_text, fps, use_gpu=False,
                字体名称="微软雅黑", 字体大小=13, 描边宽度=1, 阴影宽度=1, 底部边距=40,
                竖屏每行最大字符数=12, 横屏每行最大字符数=20):
        print(f"\n{'='*60}")
        print(f"[VideoSubtitleFromImages] 开始处理 (FFmpeg drawtext模式)")
        print(f"  字体: {字体名称}, 大小: {字体大小}")
        print(f"  描边: {描边宽度}, 阴影: {阴影宽度}, 底部边距: {底部边距}")
        print(f"  竖屏最大字符: {竖屏每行最大字符数}, 横屏最大字符: {横屏每行最大字符数}")
        print(f"  GPU加速: {'开启' if use_gpu else '关闭'}")
        print(f"{'='*60}")
        
        if not subtitle_text:
            print("[VideoSubtitleFromImages] 字幕文本为空，返回原图")
            return (images,)
        
        start_time = time.time()
        
        output_dir = OUTPUT_DIR
        timestamp = int(time.time())
        temp_video = os.path.join(output_dir, f"temp_{timestamp}.mp4")
        subtitle_video = os.path.join(output_dir, f"subtitle_{timestamp}.mp4")
        frames_dir = os.path.join(output_dir, f"frames_{timestamp}")
        
        print(f"[VideoSubtitleFromImages] 步骤1: 将图像帧转为临时视频...")
        temp_result = tensor_to_video(images, temp_video, fps, use_gpu)
        
        if not temp_result or not os.path.exists(temp_video):
            print(f"[VideoSubtitleFromImages] 临时视频创建失败")
            return (images,)
        
        print(f"[VideoSubtitleFromImages] 步骤2: 使用FFmpeg drawtext渲染字幕...")
        
        global font_size, font_name, style, max_chars_config
        old_font_size = font_size
        old_font_name = font_name
        old_style = style.copy()
        old_max_chars = max_chars_config.copy()
        
        font_size = 字体大小
        font_name = 字体名称
        style["outline"] = 描边宽度
        style["shadow"] = 阴影宽度
        style["margin_v"] = 底部边距
        max_chars_config["vertical"] = 竖屏每行最大字符数
        max_chars_config["horizontal"] = 横屏每行最大字符数
        
        try:
            result = process_subtitle(temp_video, subtitle_text, subtitle_video, use_gpu)
            
            font_size = old_font_size
            font_name = old_font_name
            style = old_style
            max_chars_config = old_max_chars
            
            if os.path.exists(temp_video):
                os.remove(temp_video)
            
            if result and os.path.exists(result):
                print(f"[VideoSubtitleFromImages] 步骤3: 从视频提取帧...")
                frames = video_to_frames(result, frames_dir)
                
                if os.path.exists(result):
                    os.remove(result)
                
                if frames:
                    tensor = frames_to_tensor(frames)
                    
                    for f in frames:
                        if os.path.exists(f):
                            os.remove(f)
                    if os.path.exists(frames_dir):
                        os.rmdir(frames_dir)
                    
                    elapsed_time = time.time() - start_time
                    print(f"\n[VideoSubtitleFromImages] 处理完成!")
                    print(f"  输出形状: {tensor.shape}")
                    print(f"  处理耗时: {elapsed_time:.2f} 秒")
                    return (tensor,)
                else:
                    print(f"[VideoSubtitleFromImages] 帧提取失败")
                    return (images,)
            else:
                print(f"[VideoSubtitleFromImages] FFmpeg处理失败")
                return (images,)
                
        except Exception as e:
            font_size = old_font_size
            font_name = old_font_name
            style = old_style
            max_chars_config = old_max_chars
            
            print(f"[VideoSubtitleFromImages] 处理异常: {str(e)}")
            import traceback
            traceback.print_exc()
            
            if os.path.exists(temp_video):
                os.remove(temp_video)
            if os.path.exists(subtitle_video):
                os.remove(subtitle_video)
            return (images,)


NODE_CLASS_MAPPINGS = {
    "VideoSubtitleNode": VideoSubtitleNode,
    "VideoSubtitleFromImages": VideoSubtitleFromImages,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoSubtitleNode": "Video Subtitle (OSS URL)",
    "VideoSubtitleFromImages": "Video Subtitle (From Images)",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

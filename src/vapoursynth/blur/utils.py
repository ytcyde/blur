import vapoursynth as vs
from vapoursynth import core

from pathlib import Path
from fractions import Fraction
from typing import Callable
import weakref


class BlurException(Exception):
    pass


def load_plugins(extension: str):
    plugin_dir = Path("../vapoursynth-plugins")
    ignored = {
        f"libbestsource{extension}",
    }

    for plugin in plugin_dir.glob(f"*{extension}"):
        if plugin.name not in ignored:
            print("Loading", plugin.name)
            try:
                core.std.LoadPlugin(path=str(plugin))
            except Exception as e:
                print(f"Failed to load plugin {plugin.name}: {e}")


def safe_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def coalesce(value, fallback):
    return fallback if value is None else value


def check_model_path(rife_model_path: str):
    path = Path(rife_model_path)

    if not path.exists():
        raise BlurException(f"RIFE model not found at {path}")


def assume_scaled_fps(clip, timescale):
    new_fps = clip.fps * timescale
    fps_frac = Fraction(new_fps).limit_denominator()

    return core.std.AssumeFPS(
        clip, fpsnum=fps_frac.numerator, fpsden=fps_frac.denominator
    )


def has_blurflow():
    if hasattr(core, "blurflow"):
        return True

    # not in vapoursynth's plugin folder, it may be sitting next to the scripts instead
    for plugin in Path(__file__).parent.parent.glob("blurflow.*"):
        try:
            core.std.LoadPlugin(path=str(plugin))
            return True
        except vs.Error:
            pass

    return False


# how to get clips that can be had as yuv420p8 without converting them, for as long as they're around
subsampled_clips: dict[int, Callable[[], vs.VideoNode | None]] = {}


def provide_subsampled(video: vs.VideoNode, get: Callable[[], vs.VideoNode | None]):
    key = id(video)
    subsampled_clips[key] = get
    weakref.finalize(video, subsampled_clips.pop, key, None)


def subsampled(video: vs.VideoNode):
    # what blurflow.Subsample would make of the clip
    key = id(video)

    if key in subsampled_clips:
        converted = subsampled_clips[key]()
        if converted is not None:
            # there's only need to get it once
            subsampled_clips[key] = lambda: converted
            return converted

    return core.blurflow.Subsample(video)


def convert_format(video: vs.VideoNode, is_full_color_range: bool, target_format):
    if video.format.id == target_format:
        return video

    # svp works in yuv420p8, so the usual conversion is to it and back from yuv444p8. blurflow
    # does just that, giving the same frames as resize.Point a lot faster
    if (
        not is_full_color_range
        and video.width % 2 == 0
        and video.height % 2 == 0
        and has_blurflow()
    ):
        match (video.format.id, target_format):
            case (vs.YUV444P8, vs.YUV420P8):
                return subsampled(video)
            case (vs.YUV420P8, vs.YUV444P8):
                return core.blurflow.Upsample(video)

    convert_kwargs = {
        "format": target_format,
        "range_in": is_full_color_range,
        "range": is_full_color_range,
    }

    target_family = core.get_video_format(target_format).color_family
    if target_family == vs.RGB and video.format.color_family == vs.YUV:
        convert_kwargs["matrix_in_s"] = "709"
    elif target_family == vs.YUV and video.format.color_family == vs.RGB:
        convert_kwargs["matrix_s"] = "709"

    return core.resize.Point(video, **convert_kwargs)


def with_format(
    video: vs.VideoNode,
    is_full_color_range: bool,
    target_format,
    process_func,
    restore_format=True,
):
    orig_format = video.format.id

    video = convert_format(video, is_full_color_range, target_format)
    video = process_func(video)

    if restore_format:
        video = convert_format(video, is_full_color_range, orig_format)

    return video

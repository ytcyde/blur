# credit to InterFrame - https://www.spirton.com/uploads/InterFrame/InterFrame2.html and https://github.com/HomeOfVapourSynthEvolution/havsfunc

import vapoursynth as vs
from vapoursynth import core

import json
import math

import blur.utils as u

LEGACY_PRESETS = ["weak", "film", "smooth", "animation"]
NEW_PRESETS = ["default", "test"]

DEFAULT_PRESET = "weak"
DEFAULT_ALGORITHM = 13
DEFAULT_BLOCKSIZE = 8
DEFAULT_OVERLAP = 2
DEFAULT_SPEED = "medium"
DEFAULT_MASKING = 50
DEFAULT_GPU = True


def generate_svp_strings(
    new_fps,
    preset=DEFAULT_PRESET,
    algorithm=DEFAULT_ALGORITHM,
    blocksize=DEFAULT_BLOCKSIZE,
    overlap=DEFAULT_OVERLAP,
    speed=DEFAULT_SPEED,
    masking=DEFAULT_MASKING,
    gpu=DEFAULT_GPU,
    scene_detect=False,
):
    # build super json
    super_json = {
        "pel": 1,
        "gpu": gpu,
    }

    # build vectors json
    vectors_json = {
        "block": {
            "w": blocksize,
            "overlap": overlap,
        }
    }

    match preset:
        case "test":
            vectors_json["main"] = {
                "search": {"type": 3, "satd": True, "coarse": {"type": 3}}
            }
        case _ if preset in LEGACY_PRESETS:
            vectors_json["main"] = {"search": {"distance": 0, "coarse": {}}}

            if preset == "weak":
                vectors_json["main"]["search"]["coarse"] = {
                    "distance": -1,
                    "trymany": True,
                    "bad": {"sad": 2000},
                }
            else:
                vectors_json["main"]["search"]["coarse"] = {"distance": -10}

    # build smooth json
    smooth_json = {
        "rate": {"num": int(new_fps), "abs": True},
        "algo": algorithm,
        "mask": {
            "area": masking,
            "area_sharp": 1.2,  # test if this does anything
        },
    }

    if not scene_detect:
        # dont want any scene detection stuff when normally blurring (i think?)
        smooth_json["scene"] = {
            "blend": False,
            "mode": 0,
            "limits": {"blocks": 9999999},
        }

    return [json.dumps(obj) for obj in [super_json, vectors_json, smooth_json]]


def svp(
    _video: vs.VideoNode,
    is_full_color_range: bool,
    super_string: str,
    vectors_string: str,
    smooth_str: str,
    restore_format=True,
    static_threshold: float | None = None,
):
    def process(video):
        super = core.svp1.Super(video, super_string)
        vectors = core.svp1.Analyse(super["clip"], super["data"], video, vectors_string)
        interpolated = core.svp2.SmoothFps(
            video,
            super["clip"],
            super["data"],
            vectors["clip"],
            vectors["data"],
            smooth_str,
        )

        if static_threshold is None:
            return interpolated

        return hold_static_intervals(video, interpolated, static_threshold)

    return u.with_format(
        _video, is_full_color_range, vs.YUV420P8, process, restore_format
    )


def hold_static_intervals(
    video: vs.VideoNode, interpolated: vs.VideoNode, threshold: float
):
    # when two consecutive source frames are duplicates there's no motion to interpolate between
    # them, so the frames in between are taken from the nearest source frame rather than rendered.
    # rendering is by far the slowest part of the pipeline, so this is a big win on static footage
    factor = interpolated.fps / video.fps
    if factor.denominator != 1 or factor < 2:
        return interpolated

    factor = int(factor)
    if len(interpolated) > len(video) * factor:
        return interpolated

    next_video = video[1:] + video[-1]
    differences = core.std.PlaneStats(video, next_video)

    first_half = (factor + 1) // 2
    held = core.std.Interleave(
        [video] * first_half + [next_video] * (factor - first_half)
    )
    differences = core.std.Interleave([differences] * factor)

    return core.akarin.Select(
        [interpolated, held[: len(interpolated)]],
        differences[: len(interpolated)],
        f"x.PlaneStatsDiff {threshold} < 1 0 ?",
    )


def interpolate_svp(
    video: vs.VideoNode,
    is_full_color_range: bool,
    new_fps: int,
    preset=DEFAULT_PRESET,
    algorithm=DEFAULT_ALGORITHM,
    blocksize=DEFAULT_BLOCKSIZE,
    overlap=DEFAULT_OVERLAP,
    speed=DEFAULT_SPEED,
    masking=DEFAULT_MASKING,
    gpu=DEFAULT_GPU,
    restore_format=True,
    static_threshold: float | None = None,
):
    preset = preset.lower()

    if preset not in LEGACY_PRESETS and preset not in NEW_PRESETS:
        raise vs.Error(f"interpolate: '{preset}' is not a valid preset")

    # generate svp strings
    [super_string, vectors_string, smooth_string] = generate_svp_strings(
        new_fps, preset, algorithm, blocksize, overlap, speed, masking, gpu
    )

    return svp(
        video,
        is_full_color_range,
        super_string,
        vectors_string,
        smooth_string,
        restore_format,
        static_threshold,
    )


def change_fps(clip, fpsnum, fpsden=1):  # this is just directly from havsfunc
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error("ChangeFPS: This is not a clip")

    factor = (fpsnum / fpsden) * (clip.fps_den / clip.fps_num)
    length = math.floor(len(clip) * factor)

    # when the source frames picked are evenly spaced (e.g. 1200 -> 60 fps), select them natively
    # instead of building a new node per frame in python, which serialises on the GIL
    indices = [math.floor(n / factor) for n in range(length)]
    if length > 1:
        step = indices[1] - indices[0]
        if step > 0 and all(b - a == step for a, b in zip(indices, indices[1:])):
            selected = clip[indices[0] :: step][:length]
            return core.std.AssumeFPS(selected, fpsnum=fpsnum, fpsden=fpsden)

    def frame_adjuster(n):
        real_n = math.floor(n / factor)
        one_frame_clip = clip[real_n] * (len(clip) + 100)
        return one_frame_clip

    attribute_clip = clip.std.BlankClip(
        length=math.floor(len(clip) * factor), fpsnum=fpsnum, fpsden=fpsden
    )
    return attribute_clip.std.FrameEval(eval=frame_adjuster)


def interpolate_mvtools(
    clip,
    new_fps,
    blocksize=DEFAULT_BLOCKSIZE,
    masking=100,
    pel=1,
    sharp=0,
    overlap=DEFAULT_OVERLAP,
    search=5,
    searchparam=3,
    pelsearch=1,
    dct=3,
    blend=False,
):
    super = core.mv.Super(
        clip, hpad=blocksize, vpad=blocksize, pel=pel, rfilter=1, sharp=sharp
    )

    analyse_args = dict(
        blksize=blocksize,
        overlap=overlap,
        search=search,
        searchparam=searchparam,
        pelsearch=pelsearch,
        dct=dct,
    )

    bv = core.mv.Analyse(super, isb=True, **analyse_args)
    fv = core.mv.Analyse(super, isb=False, **analyse_args)

    return core.mv.FlowFPS(
        clip, super, bv, fv, num=new_fps, den=1, blend=blend, ml=max(masking, 1)
    )


def interpolate_rife(
    _video: vs.VideoNode,
    is_full_color_range: bool,
    new_fps: int,
    model_path: str,
    gpu_index: int,
):
    u.check_model_path(model_path)

    def process(video):
        return core.rife.RIFE(
            video,
            fps_num=new_fps,
            fps_den=1,
            model_path=model_path,
            gpu_id=gpu_index,
        )

    return u.with_format(_video, is_full_color_range, vs.RGBS, process)

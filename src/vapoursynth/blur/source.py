import vapoursynth as vs
from vapoursynth import core

import shutil
import subprocess
from pathlib import Path

import blur.utils as u

# what the gpu decoder takes, and the format it comes out as. it can decode h264 as well, but
# nvidia's decoder doesn't give quite the frames a software decoder does for some h264 video
GPU_CODECS = ["hevc"]
GPU_FORMATS = {
    "yuv420p": vs.YUV420P8,
    "yuvj420p": vs.YUV420P8,
    "yuv444p": vs.YUV444P8,
    "yuvj444p": vs.YUV444P8,
}


def find_tool(name: str):
    # bundled with blur, next to the scripts
    for bundled in Path(__file__).parent.parent.glob(f"ffmpeg/{name}*"):
        return str(bundled)

    return shutil.which(name)


def run(command: list[str]):
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


def cfr_frame_map(
    timestamps: list[float],
    discarded: list[bool],
    num_frames: int,
    fps_num: int,
    fps_den: int,
):
    # which of the video's frames (counting in the order they're shown in) each frame of a
    # constant framerate clip is. follows what LWLibavSource does when it's given a framerate, so
    # that the clip is the same as its, frame for frame
    frame_map = []
    candidate = 0

    for n in range(num_frames):
        target = (n * fps_den) / fps_num
        next_target = ((n + 1) * fps_den) / fps_num

        # first frame at or after the time wanted
        while candidate < len(timestamps) and timestamps[candidate] < target:
            candidate += 1

        if candidate >= len(timestamps):
            frame_map.append(len(timestamps) - 1)
        elif candidate == 0:
            frame_map.append(0)
        else:
            current = timestamps[candidate]
            previous = timestamps[candidate - 1]

            # it's the frame for this time if it's nearer to it than both the frame before it and
            # the next time are. otherwise the frame before it is still showing
            if (
                current > next_target
                or current > (next_target + target) / 2
                or current - target >= target - previous
            ):
                frame_map.append(candidate - 1)
            else:
                frame_map.append(candidate)

    # frames the container marks as discarded (trimmed off by an edit list) aren't shown. the
    # next frame that is stands in for them
    shown = [frame for frame, is_discarded in enumerate(discarded) if not is_discarded]
    if not shown:
        return None

    stand_in = []
    next_shown = 0
    for frame in range(len(discarded)):
        if next_shown < len(shown) - 1 and shown[next_shown] < frame:
            next_shown += 1
        stand_in.append(shown[next_shown])

    return [stand_in[frame] for frame in frame_map]


def gpu_decoded(reference: vs.VideoNode, video_path: Path, fps_num: int, fps_den: int):
    # the same clip as `reference`, decoded by the gpu's own decoder, or None if that can't be
    # done. a regular source filter decodes on the cpu, in competition with everything else that
    # runs there, and ends up being what the rest of the pipeline waits on
    if fps_num <= 0 or fps_den <= 0:
        return None

    if not u.has_blurflow() or not hasattr(core.blurflow, "Source"):
        return None

    ffmpeg = find_tool("ffmpeg")
    ffprobe = find_tool("ffprobe")
    if not ffmpeg or not ffprobe:
        return None

    probe = run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,pix_fmt,time_base:packet=pts,flags",
            "-of",
            "csv=nokey=0",
            str(video_path),
        ]
    )
    if probe.returncode != 0:
        return None

    stream = {}
    packets = []  # in the order they're decoded in

    try:
        for line in probe.stdout.splitlines():
            kind, *fields = line.split(",")
            values = dict(field.split("=", 1) for field in fields if "=" in field)

            if kind == "packet":
                packets.append((int(values["pts"]), "D" in values["flags"]))
            elif kind == "stream":
                stream = values

        codec = stream["codec_name"]
        pix_fmt = stream["pix_fmt"]
        time_base_num, time_base_den = map(int, stream["time_base"].split("/"))
    except (KeyError, ValueError):
        return None

    if not packets or time_base_den <= 0:
        return None

    if codec not in GPU_CODECS or GPU_FORMATS.get(pix_fmt) != reference.format.id:
        return None

    # the order the frames are shown in
    shown_order = sorted(range(len(packets)), key=lambda packet: packets[packet][0])
    timestamps = [packets[packet][0] for packet in shown_order]
    if any(a == b for a, b in zip(timestamps, timestamps[1:])):
        return None

    display_order = [0] * len(packets)
    for position, packet in enumerate(shown_order):
        display_order[packet] = position

    frame_map = cfr_frame_map(
        [(pts - timestamps[0]) * time_base_num / time_base_den for pts in timestamps],
        [packets[packet][1] for packet in shown_order],
        len(reference),
        fps_num,
        fps_den,
    )
    if frame_map is None:
        return None

    # the decoder takes the video as a plain stream, which ffmpeg copies out of the container
    command = [
        ffmpeg,
        "-loglevel",
        "error",
        "-nostdin",
        "-i",
        str(video_path),
        "-map",
        "0:v:0",
        "-c",
        "copy",
        "-bsf:v",
        f"{codec}_mp4toannexb",
        "-f",
        codec,
        "-",
    ]

    try:
        return core.blurflow.Source(
            reference,
            command=command,
            frames=frame_map,
            order=display_order,
            codec=codec,
        )
    except vs.Error:
        return None


def open_video(video_path: Path, fps_num, fps_den, gpu_decoding: bool):
    # vspipe passes script arguments as strings
    fps_num = int(fps_num)
    fps_den = int(fps_den)

    def lsmash(prefer_hw: int):
        return core.lsmas.LWLibavSource(
            source=video_path,
            cache=0,
            prefer_hw=prefer_hw,
            fpsnum=fps_num if fps_num != -1 else None,
            fpsden=fps_den if fps_den != -1 else None,
        )

    video = lsmash(0)
    if not gpu_decoding:
        return video

    return gpu_decoded(video, video_path, fps_num, fps_den) or lsmash(3)

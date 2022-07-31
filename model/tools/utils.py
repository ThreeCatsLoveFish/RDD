import os
import subprocess

class VideoWriter:
    def __init__(self, filename, fps=24, use_moviepy=False) -> None:
        self.filename = filename
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.fps = fps
        self.p = None
        self.shape = None

        # moviepy is not recommended, 
        # please use ffmpeg instead whenever possible.
        self.use_moviepy = use_moviepy
        if self.use_moviepy:
            self.frames = []

    
    def write(self, frame):
        if not self.filename:
            return
        if self.use_moviepy:
            self.frames.append(frame)
        else:
            if self.p is None:
                h, w, _ = self.shape = frame.shape
                self.p = subprocess.Popen([
                    "/usr/bin/ffmpeg",
                    '-y',  # overwrite output file if it exists
                    '-f', 'rawvideo',
                    '-vcodec','rawvideo',
                    '-s', f'{w}x{h}',  # size of one frame
                    '-pix_fmt', 'rgb24',
                    '-r', f'{self.fps}',  # frames per second
                    '-i', '-',  # The imput comes from a pipe
                    '-an',  # Tells FFMPEG not to expect any audio
                    '-loglevel', 'error',
                    '-pix_fmt', 'yuv420p',
                    '-crf', '18',
                    '-tune', 'fastdecode',
                    # '-x264opts', 'keyint=0:min-keyint=0:no-scenecut',
                    self.filename
                ], stdin=subprocess.PIPE)
            assert self.shape == frame.shape
            self.p.stdin.write(frame.tobytes())

    def close(self):
        if self.use_moviepy:
            import moviepy.video.io.ImageSequenceClip as ImageSequenceClip
            clip = ImageSequenceClip.ImageSequenceClip(self.frames, fps=24)
            clip.write_videofile(self.filename, threads=16)
        else:
            self.p.stdin.flush()
            self.p.stdin.close()
            return self.p.wait()

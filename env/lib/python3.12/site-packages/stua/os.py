# Copyright (C) Raffaele Salmaso <raffaele@salmaso.org>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from contextlib import contextmanager
import os
from pathlib import Path
import subprocess
import sys


@contextmanager
def cd(path):
    old_path = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_path)


def puts(*args):
    sys.stdout.write("".join([str(arg) for arg in args]))
    sys.stdout.write("\n")
    sys.stdout.flush()


def system(
    args,
    *,
    stdin=None,
    stdout=None,
    stderr=None,
    capture_output=False,
    shell=False,
    cwd=None,
    timeout=None,
    check=False,
    encoding=None,
    errors=None,
    text=None,
    env=None,
):
    kwargs = {
        "stdin": stdin,
        "stdout": stdout,
        "stderr": stderr,
        "capture_output": capture_output,
        "shell": shell,
        "cwd": cwd,
        "timeout": timeout,
        "check": check,
        "encoding": encoding,
        "errors": errors,
        "text": text,
        "env": env,
    }
    return subprocess.run([args] if isinstance(args, str) else list(args), **kwargs)


def mkdir(path, *, mode=511, parents=True, exist_ok=True):
    """
    create a directory
    """
    path = path if isinstance(path, Path) else Path(path)
    path.mkdir(mode=mode, parents=parents, exist_ok=exist_ok)

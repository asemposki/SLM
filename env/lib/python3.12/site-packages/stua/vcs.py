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

import datetime
import os
import subprocess


def get_hg_changeset(filename=None):
    filename = __file__ if filename is None else filename
    hg_log = subprocess.Popen(
        "hg parent --template '{date|shortdate}_{node|short}'",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(filename))),
        universal_newlines=True,
    )
    return hg_log.communicate()[0].replace("-", "").replace("_", "-")


def get_git_changeset(filename=None):
    filename = __file__ if filename is None else filename
    git_log = subprocess.Popen(
        "git log --pretty=format:%ct --quiet -1 HEAD",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(filename))),
        universal_newlines=True,
    )
    timestamp = git_log.communicate()[0]
    try:
        timestamp = datetime.datetime.utcfromtimestamp(int(timestamp))
    except ValueError:
        return None
    return timestamp.strftime("%Y%m%d%H%M%S")

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

from .vcs import get_git_changeset, get_hg_changeset

HG = "hg"
GIT = "git"
NONE = ""


def get_version(version=None, vcs=NONE, filename=None):
    "Returns a PEP 386-compliant version number from VERSION."

    if version is None:
        from . import VERSION as version
    else:
        assert len(version) == 5
        assert version[3] in ("alpha", "beta", "rc", "final")

    parts = 2 if version[2] == 0 else 3
    main = ".".join(str(x) for x in version[:parts])

    sub = ""
    if version[3] == "alpha" and version[4] == 0 and vcs:
        changeset = {"hg": get_hg_changeset, "git": get_git_changeset}[vcs](filename)
        if changeset:
            sub = ".dev{}".format(changeset)

    elif version[3] != "final":
        mapping = {"alpha": "a", "beta": "b", "rc": "rc"}
        sub = mapping[version[3]] + str(version[4])

    return str(main + sub)

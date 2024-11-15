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

from argparse import ArgumentParser
import os
import sys

from . import version


class BaseCommand:
    """
    Base Command class
    """

    def handle(self, command, args):
        """
        The actual logic of the command. Subclasses must implement this method.
        """
        raise NotImplementedError("subclasses of BaseCommand must provide a handle(command, args) method")

    def execute(self, command, args):
        return self.handle(command, args=args)

    def __init__(self, stdout=None, stderr=None):
        self.stdout = stdout or sys.stdout
        self.stderr = stderr or sys.stderr

    def run(self, argv):
        command, args = argv[0], argv[1:]
        status = self.execute(command, args)
        return 0 if status is None else status


class Command(BaseCommand):
    """"""

    add_help = True
    help = ""

    def handle(self, command, options):
        """
        The actual logic of the command. Subclasses must implement this method.
        """
        raise NotImplementedError("subclasses of Command must provide a handle(command, options) method")

    def get_version(self):
        return version.get_version()

    def create_parser(self, prog_name):
        return ArgumentParser(prog=os.path.basename(prog_name), add_help=self.add_help, description=self.help or None)

    def default_arguments(self, parser):
        parser.add_argument("--version", action="version", version=self.get_version())

    def add_arguments(self, parser):
        pass

    def execute(self, command, args):
        parser = self.create_parser(command)
        self.default_arguments(parser)
        self.add_arguments(parser)
        return self.parse(parser, command, args)

    def parse(self, parser, command, args):
        options = parser.parse_args(args)
        return self.handle(command, options)


class ArgumentCommand(Command):
    """"""

    def handle(self, command, options, args):
        """
        The actual logic of the command. Subclasses must implement this method.
        """
        raise NotImplementedError("subclasses of ArgumentCommand must provide a handle(command, options, args) method")

    def parse(self, parser, command, args):
        options, args = parser.parse_known_args(args)
        return self.handle(command, options, args)

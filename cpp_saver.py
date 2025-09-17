import subprocess
import os
from IPython.core.magic import Magics, cell_magic, magics_class
from IPython.core.magic_arguments import argument, magic_arguments, parse_argstring
from pygments import highlight
from pygments.lexers import CppLexer
from pygments.formatters import HtmlFormatter
from IPython.display import display, HTML

def print_out(out: str):
    for l in out.split('\n'):
        print(l)

def displayHTML(html_code):
    '''
    Display HTML in notebook
    '''
    display(HTML(html_code))

@magics_class
class CPP(Magics):
    @staticmethod
    def compile(src, out):
        compiler = 'g++'
        res = subprocess.check_output(
            [compiler, "-o", out, src], stderr=subprocess.STDOUT)
        print_out(res.decode("utf8"))

    @staticmethod
    def custom_compile(arg_list):
        res = subprocess.check_output(
            arg_list, stderr=subprocess.STDOUT)
        print_out(res.decode("utf8"))

    @magic_arguments()
    @argument('-n', '--name', type=str, help='File name that will be produced by the cell.')
    @argument('-c', '--compile', type=str, help='Compile command. Use true for default command or specify command in single quotes.')
    @argument('-a', '--append', help='Should be appended to same file', action="store_true")
    @argument('-s', '--style', type=str, help='Pygments style name')
    @cell_magic
    def cpp(self, line='', cell=None):
        '''
        C++ syntax highlighting cell magic.
        '''
        global style
        args = parse_argstring(self.cpp, line)
        if args.name != None:
            ex = args.name.split('.')[-1]
            if ex not in ['c', 'cpp', 'h', 'hpp']:
                raise Exception('Name must end with .cpp, .c, .hpp, or .h')
        else:
            args.name = 'src.cpp'

        if args.append:
            mode = "a"
        else:
            mode = "w"
    
        with open(args.name, mode) as f:
            f.write(cell)

        if args.compile != None:
            try:
                if args.compile == 'true':
                    self.compile(args.name, args.name.split('.')[0])
                else:
                    self.custom_compile(args.compile.replace("'", "").split(' '))
            except subprocess.CalledProcessError as e:
                    print_out(e.output.decode("utf8"))

        if args.style != None:
            displayHTML(highlight(cell, CppLexer(), HtmlFormatter(full=True,nobackground=True,style=args.style)))

def load_ipython_extension(ip):
    os.system('pip install pygments ipywidgets')
    plugin = CPP(ip)
    ip.register_magics(plugin)


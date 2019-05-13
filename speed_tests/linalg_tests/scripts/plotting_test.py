from .plotting import *

if __name__ == '__main__':
    parser = create_parser()
    namespace = parser.parse_args()
    plotter = Plotter(namespace.path)
    plotter.create_figs()

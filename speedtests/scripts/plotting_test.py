from ..plotting import *

if __name__ == '__main__':
    parser = create_parser()
    namespace = parser.parse_args()
    plotter = Plotter(namespace.path)
    print('================================================')
    print(plotter.datasets_count)
    print('================================================')
    plotter.datasets_printer(0)
    print('================================================')
    plotter.prepare_data()
    print('================================================')
    plotter.create_figs()

import sys, getopt
from survey import *
import pickle


def help(exit_code):
    print('Usage:')
    print('main.py {--train | --test} --modelfile=<filename> [options]')
    print(' -h                         :   print this help message.')
    print(' --train                    :   run the training. datafile option is required.')
    print(' --test=<survey row number> :   run the test for specific survey. Training should be done before calling test.')
    print(' -m / --modelfile=          :   defines the excel data file to be read.')
    print(' -d / --datafile=           :   defines the excel data file to be read.')
    print(' -v / --verbosity=          :   sets the verbosity level.')
    print('')
    print('')

    sys.exit(exit_code)


def main(argv):
    data_file = ''
    model_file = ''
    verbosity = 0
    mode = None
    test_row = None


    try:
        opts, args = getopt.getopt(argv, "v:hd:", ["test=", "train", "modelfile=", "datafile=", "verbosity="])
    except getopt.GetoptError:
        help(2)

    for opt, arg in opts:
        if opt == '-h':
            help(0)
        elif opt == "--test":
            mode = 1
            test_row = int(arg)
        elif opt == "--train":
            mode = 2
        elif opt in ("-d", "--datafile"):
            data_file = arg
        elif opt in ("-v", "--verbosity"):
            verbosity = int(arg)
        elif opt in ("-m", "--modelfile"):
            model_file = arg

    assert mode != 0, "ERROR 1: Either --test or --train is required."
    assert model_file != '', "ERROR 2: --model_file is required."
    assert mode != 2 or data_file != '' , "ERROR 3: In --train mode, --datafile is required."

    # data_file = 'C:\\Users\\Mahta\\PycharmProjects\\TR_assessment\\sample data.csv'
    # Train Mode
    if mode == 2:
        ss = survey_set(verbosity=verbosity)
        ss.import_data(data_file)
        ss.classify()

        # Store data (serialize)
        with open(model_file, 'wb') as handle:
            pickle.dump(ss, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Test
    elif mode == 1:
        # Load data (deserialize)
        with open(model_file, 'rb') as handle:
            ss = pickle.load(handle)

        farm_label = ss.classifier.model.predict([ss.metrics[test_row]])
        if farm_label[0] == 1:
            ss.sset[test_row].fix_overlapping_borders()
            ss.sset[test_row].draw()
        if farm_label[0] == 2:
            ss.sset[test_row].fix_similar_borders()
            ss.sset[test_row].draw()
        else:
            ss.sset[test_row].draw()


if __name__ == "__main__":
    main(sys.argv[1:])


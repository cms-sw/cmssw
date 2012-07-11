from datetime import datetime
from optparse import OptionParser
from ROOT import TFile

def find_file_names(directory, names_list):
    for key in directory.GetListOfKeys():
        subdir = directory.Get(key.GetName())
        if subdir:
            if subdir.IsFolder():
                find_file_names(subdir, names_list)
            else:
                filename = directory.GetPath().split(':')[1] + ': ' + subdir.GetName()
                names_list.add(filename)

def dqm_diff(filename1, filename2):
    print "Missing files:"
    f1 = TFile(filename1)
    f1_dir = f1.GetDirectory("DQMData")
    f1_file_names = set()
    find_file_names(f1_dir, f1_file_names)
    f1.Close()
    f2 = TFile(filename2)
    f2_dir = f2.GetDirectory("DQMData")
    f2_file_names = set()
    find_file_names(f2_dir, f2_file_names)
    f2.Close()
    printed = False
    for name in f1_file_names:
        if name not in f2_file_names:
            print "  ->", name
            printed = True
    if not printed:
        print "    All files match."

parser = OptionParser(usage='usage: %prog <root_file1> <root_file2>')
(options, args) = parser.parse_args()
if len(args) != 2:
    parser.error("You have to specify two root files. e.g. ``dqm_diff.py file1.root file2.root``.")

dqm_diff(*args)

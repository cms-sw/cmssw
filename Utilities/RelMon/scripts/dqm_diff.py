#! /usr/bin/env python3
'''
Script prints out histogram names that are in one ROOT file but not in another.

Author:  Albertas Gimbutas,  Vilnius University (LT)
e-mail:  albertasgim@gmail.com
'''
from __future__ import print_function
from datetime import datetime, timedelta
from optparse import OptionParser

def collect_directory_filenames(directory, names_list):
    """Adds current directory file (histogram) names to ``names_list``. Then
    recursively calls itself for every current directory sub-directories."""
    for key in directory.GetListOfKeys():
        subdir = directory.Get(key.GetName())
        if subdir:
            if subdir.IsFolder():
                collect_directory_filenames(subdir, names_list)
            else:
                filename = directory.GetPath().split(':')[1] + ': ' + subdir.GetName()
                names_list.add(filename)

def get_content(root_file_name):
    """Returns all file (histogram) names, which are found in <root_file_name>."""
    from ROOT import TFile
    root_file = TFile(root_file_name)
    root_directory = root_file.GetDirectory("DQMData")
    filename_set = set()
    collect_directory_filenames(root_directory, filename_set)
    root_file.Close()
    return filename_set

def dqm_diff(filename1, filename2):
    """Prints file (histogram) names that are in <file1> and not in <file2>."""
    print("Missing files:")
    content1 = get_content(filename1)
    content2 = get_content(filename2)
    printed = False
    for name in content1:
        if name not in content2:
            print("  ->", name)
            printed = True
    if not printed:
        print("    All files match.")


## Define commandline options
parser = OptionParser(usage='usage: %prog <root_file1> <root_file2> [options]')
parser.add_option('-t', '--time', action='store_true', default=False,
                    dest='show_exec_time', help='Show execution time.')
(options, args) = parser.parse_args()

## Check for commandline option errors
if len(args) != 2:
    parser.error("You have to specify two root files. e.g. ``dqm_diff.py file1.root file2.root``.")

## Execute the search of dismatches in two root fies.
start = datetime.now()
dqm_diff(*args)
if options.show_exec_time:
    print('Execution time:', str(timedelta(seconds=(datetime.now() - start).seconds)))

#! /usr/bin/env python3
"""
The script compares two ROOT files and fills specified database file with
comparison information.

Author:  Albertas Gimbutas,  Vilnius University (LT)
e-mail:  albertasgim@gmail.com

Note: balcklist support not implemented.
"""
from __future__ import print_function
import sys
import sqlite3
from datetime import datetime
from multiprocessing import Pool, Queue, Process
from optparse import OptionParser, OptionGroup
from os import makedirs
from os.path import basename, join, exists
from optparse import OptionParser

from Utilities.RelMon.utils_v2 import ComparisonError, tests, init_database, get_version
from Utilities.RelMon.web.app_utils import get_release, get_stats, get_percentage, get_img_url, get_dataset_name


parser = OptionParser(usage='Usage: %prog <file1> <file2> --db DB_NAME [options]')
parser.add_option('--dir', action='store', dest='dir', default='.',
        help='Directory to store static html and .db file.')
parser.add_option('--db', action='store', dest='db_name', default=None, help='path to SQLite3 database file.')
parser.add_option('--st_test', action='store', dest='st_test', default='KS',
        help='Statistical test to use for the comparison.')
parser.add_option('--th', action='store', dest='threshold', default=1e-5,
        help='Threshold to use in static HTML. Default: %default.')
parser.add_option('--html', action='store_true', dest='html', default=False,
        help='Generate static html. Default: %default.')
parser.add_option('--cl', action='store_true', dest='clear_db', default=False,
        help='Clear database before using.')

class RootFileComparison(object):
    def __init__(self, db_name, work_path=None, do_html=False):
        self.db_name = db_name
        self.work_path = work_path
        self.do_html = do_html

    def walk_through(self, c, directory, f1, f2, st_test, parent_id=None, path=''):
        c.execute('''INSERT INTO Directory(name, parent_id) VALUES (?, ?)''',
                     (directory.GetName(), parent_id))
        dir_id = c.lastrowid
        from_id, till_id = None, None
        for elem in directory.GetListOfKeys():
            elem_name = elem.GetName()
            subdir = directory.Get(elem_name)
            if subdir:
                if subdir.IsFolder():
                    subdir_from_id, subdir_till_id, subdir_id = self.walk_through(c, subdir,
                                            f1, f2, st_test, dir_id, path=join(path, elem_name))
                    if subdir_till_id and (not till_id or subdir_till_id > till_id):
                        till_id = subdir_till_id
                    if subdir_from_id and (not from_id or subdir_from_id < from_id):
                        from_id = subdir_from_id
                else:
                    hist1 = f1.Get(join(directory.GetPath(), elem_name))
                    hist2 = f2.Get(join(directory.GetPath(), elem_name))
                    try:
                        p_value = st_test.do_test(hist1, hist2)
                        c.execute('''INSERT INTO HistogramComparison(name, p_value, directory_id)
                                     VALUES (?, ?, ?)''', (elem_name, p_value, dir_id))
                        comp_id = c.lastrowid
                        if comp_id > till_id:
                            till_id = comp_id
                        if not from_id or comp_id < from_id:
                            from_id = comp_id
                    except ComparisonError as e:
                        print('Error comparing %s: %s' % (hist1, e))
        if from_id and till_id:
            c.execute('''UPDATE Directory SET from_histogram_id=?, till_histogram_id=?
                           WHERE id=?''', (from_id, till_id, dir_id))
        return from_id, till_id, dir_id

    def compare(self, filename1, filename2, st_test):
        if not 'TFile' in globals():
            from ROOT import TFile
        f1 = TFile(filename1)
        f2 = TFile(filename2)

        conn = sqlite3.connect(self.db_name)
        c = conn.cursor()

        ## Create Directory and HistogramComparison structure in the DB
        dir_DQMData = f1.GetDirectory("DQMData")
        dir_Run = None
        for elem in dir_DQMData.GetListOfKeys():
            elem_name = elem.GetName()
            if elem_name.startswith('Run '):
                dir_Run = dir_DQMData.Get(elem_name)

        fid, tid, dir_id = self.walk_through(c, dir_Run, f1, f2, st_test)

        c.execute('''DELETE FROM Directory WHERE from_histogram_id IS NULL
                     AND till_histogram_id IS NULL''')
        c.execute('''INSERT INTO RootFileComparison(filename1, filename2, directory_id)
                     VALUES (?, ?, ?)''', (basename(filename1), basename(filename2), dir_id))
        root_file_comparison_id = c.lastrowid

        conn.commit()
        conn.close()
        f1.Close()
        f2.Close()
        return root_file_comparison_id

    def was_compared(self, filename1, filename2, st_test_name):
        conn = sqlite3.connect(self.db_name)
        c = conn.cursor()
        c.execute('''SELECT release_comparison_id FROM RootFileComparison WHERE (filename1=? and filename2=?)
                     OR (filename1=? and filename2=?)''', (filename1, filename2, filename2, filename1))
        file_comparison = c.fetchall()

        for release_comparison_id in file_comparison:
            c.execute('''SELECT statistical_test FROM ReleaseComparison WHERE
                         id = ?''', release_comparison_id)
            statistical_test = c.fetchone()
            if statistical_test and statistical_test[0] == st_test_name:
                conn.close()
                return True
        conn.close()
        return False


if __name__ == '__main__':
    opts, args = parser.parse_args()
    if len(args) != 2:
        parser.error('Specify two files to use for the comparison.')
    if not opts.db_name:
        parser.error('Specify SQLite3 database file for the comparison.')

    if opts.clear_db:
        print('Clearing DB: %s...' % opts.db_name, end=' ')
        open(opts.db_name, 'w').close()
        print('Done.')

    init_database(opts.db_name)

    print('Comparing files:\n%s\n%s\n' % (basename(args[0]), basename(args[1])))
    file_cmp = RootFileComparison(opts.db_name, opts.dir)
    file_cmp.compare(args[0], args[1], tests[opts.st_test]())

    if opts.html:
        dbfile2html(opts.db_name, opts.threshold)

#! /usr/bin/env python3
"""
The script compares two releases, generates SQLite3 database file with release
comparison information.

Author:  Albertas Gimbutas,  Vilnius University (LT)
e-mail:  albertasgim@gmail.com

Note: default Pool size for file comparison is 7.
Note: did NOT finish static HTML generation implementation.
"""
from __future__ import print_function
import sqlite3
from datetime import datetime
from multiprocessing import Pool, Queue, Process
from subprocess import call
from optparse import OptionParser, OptionGroup
from os import makedirs, remove
from os.path import basename, join, exists

from Utilities.RelMon.utils_v2 import *
from compare_using_files_v2 import RootFileComparison


##  Parse options
parser = OptionParser(usage='Usage: %prog --re1 RELEASE1 [--f1 FR,FR,..] ' +
                            '--re2 RELEASE2 [--f2 FR,FR,..] [--st ST_TESTS] [options]')
parser.add_option('--re1', action='store', dest='release1', default=None,
                    help='First CMSSW release for release comparison, e.g. CMSSW_5_3_2_pre7.')
parser.add_option('--re2', action='store', dest='release2', default=None,
                    help='Second CMSSW release for release comparison.')
parser.add_option('--f1', action='store', dest='fragments1', default='',
                    help='Comma separated filename fragments that have or have not to be ' +
                    'in RELEASE1 filenames. For "not include" use `!` before fragment, ' +
                    'e.g. `--f1 FullSim,!2012`.''')
parser.add_option('--f2', action='store', dest='fragments2', default='',
                    help='Comma separated filename fragments that have or have not to be ' +
                    'in RELEASE2 filenames. For "not include" use `!` before fragment.''')

optional_group = OptionGroup(parser, 'Optional')
optional_group.add_option('--st', action='store', dest='st_tests', default='KS',
                    help='Comma separated statistical tests to use. \nAvailable: KS, Chi2. Default: %default.')
optional_group.add_option('--title', action='store', dest='title', default=None,
                    help='Release comparison title.')
optional_group.add_option('--dir', action='store', dest='dir', default=None,
        help='Directory to download and compare files in.')
optional_group.add_option('--url', action='store', dest='url', default=None,
                    help='URL to fetch ROOT files from. File search is recursive ' +
                    'for links in given URL.')
optional_group.add_option('--no-url', action='store_true', dest='no_url', default=False,
                    help='Search for files in DIR (specified by --dir option), ' +
                    'do NOT browse for files online.')
optional_group.add_option('--db', action='store', dest='db_name', default=None,
        help='SQLite3 .db filename to use for the comparison. Default: auto-generated SQLite3 .db file.')
optional_group.add_option('--cl', action='store_true', dest='clear_db', default=False,
                    help='Clean DB before comparison.')
optional_group.add_option('--dry', action='store_true', dest='dry', default=False,
                    help='Do not download or compare files, just show the progress.')
optional_group.add_option('--html', action='store_true', dest='html', default=False,
                    help='Generate static html. Default: %default.')
parser.add_option_group(optional_group)


def call_compare_using_files(args):
    file1, file2, work_path, db_name, clear_db = args
    command = ['./compare_using_files_v2.py', join(work_path, file1), join(work_path, file2), '--db', db_name]
    if clear_db:
        command.append('--cl')
    return call(command)

def partial_db_name(db_name, i):
    """Generates temporary database name."""
    return '%s___%d.db' % (db_name.strip('.db'), i + 1)

def merge_dbs(main_db, partial_db):
    conn = sqlite3.connect(main_db)
    c = conn.cursor()

    ## Test if database is empty
    c.execute('''SELECT * FROM Directory limit 1;''')
    directory_row = c.fetchall()

    ## Select offsets
    rel_cmp_offset, file_cmp_offset, directory_offset, hist_cmp_offset = 0, 0, 0, 0
    if directory_row:
        c.execute('''SELECT count(*) FROM ReleaseComparison;''')
        rel_cmp_offset = c.fetchone()[0]
        c.execute('''SELECT count(*) FROM RootFileComparison;''')
        file_cmp_offset = c.fetchone()[0]
        c.execute('''SELECT max(id) FROM Directory;''')
        directory_offset = c.fetchone()[0]
        c.execute('''SELECT max(id) FROM HistogramComparison;''')
        hist_cmp_offset = c.fetchone()[0]

    ## Merge DBs
    c.executescript("""
    ATTACH '{0}' AS partial;
    BEGIN;

    INSERT INTO ReleaseComparison (title, release1, release2, statistical_test)
    SELECT title, release1, release2, statistical_test FROM partial.ReleaseComparison;

    INSERT INTO RootFileComparison (filename1, filename2, release_comparison_id, directory_id)
    SELECT filename1, filename2, release_comparison_id+{1}, directory_id+{3} FROM partial.RootFileComparison;

    INSERT INTO Directory (id, name, parent_id, from_histogram_id, till_histogram_id)
    SELECT id+{3}, name, parent_id+{3}, from_histogram_id+{4}, till_histogram_id+{4} FROM partial.Directory;

    INSERT INTO HistogramComparison (name, p_value, directory_id)
    SELECT name, p_value, directory_id+{3} FROM partial.HistogramComparison;

    COMMIT;""".format(partial_db, rel_cmp_offset, file_cmp_offset, directory_offset, hist_cmp_offset))

    ## Select Last RootFileComparison ID
    c.execute('''SELECT max(id) FROM RootFileComparison;''')
    max_file_cmp_id = c.fetchone()[0]
    conn.close()
    return max_file_cmp_id


class ReleaseComparison(object):
    """Generates release comparison information and stores it on SQLite3 .db file."""
    def __init__(self, work_path=None, db_name=None, clear_db=False, dry=False, no_url=False, use_external=False):
        self.work_path = work_path
        self.db_name = db_name
        self.clear_db = clear_db
        self.dry = dry
        self.no_url = no_url
        self.use_external_script_to_compare_files = use_external

    def was_compared(self, release1, release2, st_test_name):
        conn = sqlite3.connect(self.db_name)
        c = conn.cursor()
        c.execute('''SELECT id FROM ReleaseComparison WHERE release1=? AND
                release2=? AND statistical_test=?''', (release1, release2, st_test_name))
        release_comparison_id = c.fetchone()
        conn.close()
        if release_comparison_id:
            return release_comparison_id[0]
        return False

    def compare(self, rel1, frags1, rel2, frags2, st_tests, url=None, title=None):
        print('\n#################     Searching for files     ###################')
        if self.no_url:
            print('Searching for files on disk at %s' % (self.work_path))
            files1, files2 = search_on_disk(self.work_path, rel1, frags1, rel2, frags2)
            file_pairs = make_file_pairs(files1, files2)
        else:
            print('Searching for files online at:')
            files_with_urls1, files_with_urls2 = recursive_search_online(url, rel1, frags1, rel2, frags2)
            file_pairs = make_file_pairs(files_with_urls1, files_with_urls2)
            files_with_urls1.update(files_with_urls2)
            files1, files2 = list(zip(*file_pairs))
            paired_files_with_urls = [(file, files_with_urls1[file]) for file in files1 + files2]

            if self.dry:
                print('DRY: nothing to do. Exiting.')
                exit()

            ## Create working directory if not given.
            if not self.work_path:
                self.work_path = '%s___VS___%s' % (get_version(files1[0]), get_version(files2[0]))
                if self.db_name:
                    self.db_name = join(self.work_path, self.db_name)

            if not exists(self.work_path):
                print('\n###################      Preparing directory     ###################')
                print('Creating working directory: %s ...' % self.work_path, end=' ')
                makedirs(self.work_path)
                print('Done.')

            print('\n#################     Downloading the files     ###################')
            total_size, files_to_download = get_size_to_download(self.work_path, paired_files_with_urls)
            check_disk_for_space(self.work_path, total_size)

            ## Download needed files.
            q = Queue()
            show_status_bar.q = q
            auth_download_file.q = q
            auth_download_file.work_dir = self.work_path

            Process(target=show_status_bar, args=(total_size,)).start()
            Pool(2).map(auth_download_file, files_to_download)
            if total_size:
                print("Done.")

        ## Create database
        print('\n#################     Preparing Database     ###################')
        if not self.db_name:
            self.db_name = '%s___VS___%s.db' % (get_version(file_pairs[0][0]), get_version(file_pairs[0][1]))

        if self.clear_db:
            print('Clearing DB: %s...' % self.db_name, end=' ')
            open(join(self.work_path, self.db_name), 'w').close()
            print('Done.')

        ## Compare file pairs.
        self.db_name = init_database(join(self.work_path, self.db_name))

        # TODO: Use multiprocessing for this task.
        for st_test_name in st_tests.split(','):
            print('\n#################     Comparing Releases (%s)     ###################' % st_test_name)
            st_test = tests[st_test_name]()

            some_files_compared = False
            file_comparison_ids = []
            if self.use_external_script_to_compare_files:
                # Compare files using compare_using_files_v2.py
                arg_list = [list(pair) + [self.work_path, partial_db_name(self.db_name, i),
                                                self.clear_db] for i, pair in enumerate(file_pairs)]
                pool = Pool(7)
                pool.map(call_compare_using_files, arg_list)

                # Merge databases
                print('\n#################     Merging DBs (%s)     ###################' % st_test_name)
                for i, pair in enumerate(file_pairs):
                    tmp_db = partial_db_name(self.db_name, i)
                    print('Merging %s...' % (basename(tmp_db),), end=' ')
                    file_comparison_ids.append(merge_dbs(self.db_name, tmp_db))
                    remove(tmp_db)
                    print('Done.')
                    some_files_compared = True
            else:
                file_comparison = RootFileComparison(self.db_name)

                for file1, file2 in file_pairs:
                    # TODO: If files are not found desplay nice message.
                    # TODO: Maybe subprocces would control the unwanted reports of RootFileComparison.compare()
                    file1_path = join(self.work_path, file1)
                    file2_path = join(self.work_path, file2)

                    if not file_comparison.was_compared(file1, file2, st_test_name):
                        print("Comparing:\n%s\n%s\n" % (file1, file2))
                        file_comparison_id = file_comparison.compare(file1_path, file2_path, st_test)
                        file_comparison_ids.append(file_comparison_id)
                        some_files_compared = True
                    else:
                        print("Already compared:\n%s\n%s\n" % (file1, file2))

            ## Calculate statistics for the release.
            release1 = get_version(file_pairs[0][0])
            release2 = get_version(file_pairs[0][1])
            if some_files_compared:
                release_comparison_id = self.was_compared(release1, release2, st_test_name)
                conn = sqlite3.connect(self.db_name)
                c = conn.cursor()
                if not release_comparison_id:
                    print('Inserting release "%s  VS  %s" description.\n' % (release1, release2))
                    if not title:
                        title = "%s__VS__%s" % (release1, release2)
                    c.execute('''INSERT INTO ReleaseComparison(title, release1, release2,
                                   statistical_test) VALUES (?, ?, ?, ?)''', (title,
                                release1, release2, st_test_name))
                    release_comparison_id = c.lastrowid
                c.executemany('''UPDATE RootFileComparison SET release_comparison_id = ?
                        WHERE id == ?''', [(release_comparison_id, fid) for fid in file_comparison_ids])
                conn.commit()
                conn.close()


if __name__ == '__main__':
    start = datetime.now()
    opts, args = parser.parse_args()
    if not opts.release1 or not opts.release2:
        parser.error('Not all releases specified. Please check --re1 and --re2 options.')

    rel_cmp = ReleaseComparison(opts.dir, opts.db_name, opts.clear_db, opts.dry, opts.no_url, use_external=True)
    rel_cmp.compare(opts.release1, opts.fragments1, opts.release2,
                        opts.fragments2, opts.st_tests, opts.url, opts.title)
    if opts.html:
        print('\n#################     Generating static HTML    #################')
        print('\n  Warrning!!!  Did NOT finished the implementation. \n')
        from Utilities.RelMon.web.dbfile2html import dbfile2html
        dbfile2html(rel_cmp.db_name, opts.dir)
    print('#################     Execution time: %s    #################\n' % (datetime.now() - start,))

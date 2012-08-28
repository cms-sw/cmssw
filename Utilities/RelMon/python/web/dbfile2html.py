#! /usr/bin/python
# coding: utf-8
'''
Generates static HTML for the given database file.
Warrning!: did not finish the implementation, see TODO comment.

Author:  Albertas Gimbutas,  Vilnius University (LT)
e-mail:  albertasgim@gmail.com
'''
import sqlite3
import re
from os import listdir, makedirs, getcwd
from os.path import isfile, join, exists, dirname, basename
from app_utils import *
from optparse import OptionParser

import sys
sys.path.append(getcwd())
from Utilities.RelMon.web.jinja2 import Environment, FileSystemLoader, escape
env = Environment(loader=FileSystemLoader('templates'))  # Template directory has to exist

parser = OptionParser(usage='Usage: %prog --db PATH_TO_DB [options]')
parser.add_option('--db', action='store', dest='db_name',
        help='Absolute path to SQLite3 database file.')
parser.add_option('--th', action='store', dest='threshold', default=1e-5,
        help='Threshold to use for static HTML statistics. Default: %default.')

def create_page(path, content):
    path = join(*path)
    if not exists(dirname(path)):
        makedirs(dirname(path))
    f = open(path + '.html', 'w')
    f.write(content)
    f.close()

def dbfile2html(db_name, work_path, threshold=1e-5):
    """
    Generates static HTML from given release comparison database file.
    Algorithm: iterates through database, renders Jinja2 templates and saves
    them to static HTML files.
    """
    if not exists(db_name):
        print "\nError: SQLite3 database file does not exsits. Exitting...\n"
        exit()

    conn = sqlite3.connect(db_name)
    c = conn.cursor()

    ## Initialise working directory.
    path = join(work_path, 'static_html')
    if not exists(path):
        makedirs(path)

    global_context = {'db_name': None , 'threshold': threshold,
                                'file_id': None, 'args': [], 'kwargs': None}
    global_context['static_html'] = True
    global_context['base_path'] = work_path.strip('/')

    ## Generate DB list page
    context = global_context.copy()
    db_list_temp = env.get_template('db_list.html')
    context['db_list'] = db_list_with_releases(work_path)
    f = open(join(path, 'index.html'), 'w')
    f.write(db_list_temp.render(context))

    ## Generate ReleaseComparison html pages
    c.execute('''SELECT id, title, statistical_test FROM ReleaseComparison;''')
    releases = c.fetchall()
    rel_summary_temp = env.get_template('release_summary.html')
    dir_summary_temp = env.get_template('directory_summary.html')


    for rel_id, release_title, st_test in releases:
        context = global_context.copy()
        context.update(get_release_summary_stats(c, release_title, st_test, threshold))
        context['release_title'] = release_title
        context['st_test'] = st_test
        create_page([path, release_title, st_test], rel_summary_temp.render(context))

        ## Generate RootFileComparison html pages
        print 'Generating %s (%s) comparison pages...' % (release_title, st_test)
        c.execute('''SELECT id, directory_id FROM RootFileComparison WHERE release_comparison_id = ?;''', (rel_id,))
        for file_id, file_top_dir_id in c.fetchall():
            context['file_id'] = file_id
            context.update(get_directory_summary_stats(c, [], file_id, threshold))
            create_page([path, release_title, st_test, str(file_id)], dir_summary_temp.render(context))

            c.execute('''SELECT id FROM Directory WHERE parent_id=?''', (file_top_dir_id,))
            children_dirs = c.fetchall()

            ## Generate Directory html pages
            def create_dir_pages(c, dir_id, dir_path):
                # Generate Directory page
                c.execute('''SELECT name FROM Directory WHERE id=?''', (dir_id,))
                dir_path.append(c.fetchone()[0])
                context.update(get_directory_summary_stats(c, dir_path, file_id, threshold))
                create_page([path, release_title, st_test, str(file_id)] + dir_path, dir_summary_temp.render(context))
                # TODO: Call for subdirectories

            for children_dir in children_dirs:
                create_dir_pages(c, children_dir[0], [])
        print 'Done.'


if __name__ == '__main__':
    opts, args = parser.parse_args()
    dbfile2html(opts.db_name, dirname(opts.db_name), opts.threshold)

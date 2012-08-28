#! /usr/bin/python
# coding: utf-8
'''
CherryPy application, which enables dynamic SQLite3 database file with release
comparison information browsing. Database file can be generated with
``ValidationMatrix_v2.py`` script.

Author:  Albertas Gimbutas,  Vilnius University (LT)
e-mail:  albertasgim@gmail.com
'''
import cherrypy as cpy
import sqlite3
from os.path import isfile
from jinja2 import Environment, FileSystemLoader, escape
from app_utils import *

env = Environment(loader=FileSystemLoader('templates'))


class BrowseDB:
    """
    CherryPy application for release comparison browsing from SQLite3 database files.
    The SQLite3 database files have to placed in the same directory as this script.
    """
    @cpy.expose
    def default(self, db_name=None, release_title=None, st_test=None, file_id=None, *args, **kwargs):
        """CherryPy controller, which handles all Http requests."""
        if kwargs:
            threshold = float(kwargs['threshold'])
        else:
            threshold = None

        context = {'db_name':db_name, 'release_title':release_title,
                   'threshold': threshold, 'st_test':st_test,
                   'file_id':file_id, 'args':args, 'kwargs':kwargs}
        if not threshold:
            threshold = 1e-5

        db_list_temp = env.get_template('db_list.html')
        if not db_name:
            context['db_list'] = db_list_with_releases()
            return db_list_temp.render(context)

        if not isfile(db_name + '.db'):
            context['db_list'] = db_list_with_releases()
            context['error'] = 'Does not exist: %s.db' % db_name
            return db_list_temp.render(context)

        conn = sqlite3.connect(db_name + '.db')
        c = conn.cursor()
        if not release_title or not st_test:
            rel_list_temp = env.get_template('release_list.html')
            context['release_list'] = get_release_list(c)
            return rel_list_temp.render(context)

        if not file_id:
            rel_summary_temp = env.get_template('release_summary.html')
            context.update(get_release_summary_stats(c, release_title,
                                              st_test, threshold))
            return rel_summary_temp.render(context)

        dir_summary_temp = env.get_template('directory_summary.html')
        context.update(get_directory_summary_stats(c, args, file_id, threshold))
        return dir_summary_temp.render(context)


if __name__ == '__main__':
    cpy.quickstart(BrowseDB())

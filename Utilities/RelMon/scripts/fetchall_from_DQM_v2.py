#! /usr/bin/env python
'''
Script fetches files matching specified RegExps from DQM GUI.

Author:  Albertas Gimbutas,  Vilnius University (LT)
e-mail:  albertasgim@gmail.com
'''
################################################################################
# Change logs:
# 2012-10-22 11:31 - Checking to Download also files <1MB (like GEN samples)
# 2012-07-09 16:10 - BugFix: RELEASE has to be in selected file names.
# 2012-07-09 16:10 - Added How-To examples and command line option
# explanations for -h option.
# 2012-07-08 17:23 - Added file download in bunches.
# 2012-07-06 14:30 - Added multiprocessing for file download:
# http://docs.python.org/library/multiprocessing
# 2012-07-06 14:09 - Added new commandline options implmenetation.
# 2012-07-06 09:48 - fixed ``--data`` commandline option small bug. Now it
# does not requires to specifie its value.
################################################################################

import re
import sys
import os

from multiprocessing import Pool, Queue, Process
from Queue import Empty
from os.path import basename, isfile
from optparse import OptionParser
from urllib2 import build_opener, Request

try:
    from Utilities.RelMon.authentication import X509CertOpen
except ImportError:
    from authentication import X509CertOpen


def auth_wget(url, chunk_size=1048576):
    """Returns the content of specified URL, which requires authentication.
    If the content is bigger than 1MB, then save it to file.
    """
    opener = build_opener(X509CertOpen())
    url_file = opener.open(Request(url))
    size = int(url_file.headers["Content-Length"])

    if size < 1048576:   # if File size < 1MB
        filename = basename(url)    #still download
        readed = url_file.read()    ## and then check if its not an empty dir (parent directory)
        if filename != '':
            outfile = open(filename, 'wb')  #then write File to local system
            outfile.write(readed)
        return readed

    filename = basename(url)
    file_id = selected_files.index(filename)

    if isfile("./%s" % filename):
        print '%d. Exsits on disk. Skipping.' % (file_id +1)
        return

    print '%d. Downloading...' % (file_id +1)
    file = open(filename, 'wb')
    # progress = 0
    chunk = url_file.read(chunk_size)
    while chunk:
        file.write(chunk)
        # progress += chunk_size
        chunk = url_file.read(chunk_size)
    print '%d.  Done.' % (file_id +1)
    file.close()


## Define options
parser = OptionParser(usage='usage: %prog [options]')
parser.add_option('-d', '--data', action='store_true', dest='is_from_data',
                  help='Fetch data relvals.')
parser.add_option('-m', '--mc', action='store_false', dest='is_from_data',
                  help='Fetch Monte Carlo relvals.')
parser.add_option('-r', '--release', action='store', dest='release',
                  help='Release to fetch from. RELEASE format "CMSSW_x_x_x", e.g. CMSSW_5_3_2.')
parser.add_option('-e', '--re', '--regexp', action='store', dest='regexp', default='',
        help='Comma separated regular expresions for file names. e.g. to fetch '+
        'files, which names contain "cos" or "jet" and does not contain "2010", use: '+
        '"cos,jet,^((?!2010).)*$".')
parser.add_option('--mthreads', action='store', default='3', dest='mthreads',
                  help='Number of threads for file download. Default is 3.')
parser.add_option('--dry', action='store_true', default=False, dest='dry_run',
                  help='Show files matched by regular expresion, but do not download them.')
## Parse sys.argv
(options, args) = parser.parse_args()
options.release = options.release.strip('"\'=')
options.regexp = options.regexp.strip('"\'=')

## Check for option errors
if options.is_from_data is None:
    parser.error('You have to specify the directory, use --mc for "RelVal" or ' +
                 '--data for "RelValData"')
elif options.release is None:
    parser.error('You have to specify the CMSSW release, use --release option. ' +
                 'E.g. --release CMSSW_5_3_2')
elif not options.mthreads.isdigit():
    parser.error('Bad --mthreads argument format. It has to be integer. E.g. ' +
                 '--mthreads 3')

## Use options
relvaldir = "RelVal"
if options.is_from_data:
    relvaldir = "RelValData"

release = re.findall('(CMSSW_\d*_\d*_)\d*(?:_[\w\d]*)?', options.release)
if not release:
    parser.error('No such CMSSW release found. Please check the ``--release`` commandline option value.')
releasedir = release[0] + "x"

base_url = 'https://cmsweb.cern.ch/dqm/relval/data/browse/ROOT/'
filedir_url = base_url + relvaldir + '/' + releasedir + '/'
filedir_html = auth_wget(filedir_url)

file_list_re = re.compile(r"<a href='[-./\w]*'>([-./\w]*)<")
all_files = file_list_re.findall(filedir_html)[1:]  # list of file names

options.mthreads = int(options.mthreads)
if options.mthreads > 3 or options.mthreads < 1:
    options.mthreads = 3

## Fetch the files, using multi-processing
file_res = [re.compile(r) for r in options.regexp.split(',') + [options.release]]
selected_files = [f for f in all_files if all([r.search(f) for r in file_res])]

print 'Downloading files:'
for i, name in enumerate(selected_files):
    print '%d. %s' % (i+1, name)

if not options.dry_run:
    print '\nProgress:'
    pool = Pool(options.mthreads)
    pool.map(auth_wget, [filedir_url + name for name in selected_files])

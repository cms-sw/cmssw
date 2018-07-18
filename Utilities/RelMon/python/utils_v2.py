#! /usr/bin/env python
'''
Help functions for ValidationMatrix_v2.py.

Author:  Albertas Gimbutas,  Vilnius University (LT)
e-mail:  albertasgim@gmail.com
'''
import sys
import re
import time
import sqlite3
from datetime import datetime
from multiprocessing import Pool, Queue, Process
import subprocess
from optparse import OptionParser, OptionGroup
from os import makedirs, listdir
from os.path import basename, dirname, isfile, splitext, join, exists, getsize
from Queue import Empty
from urllib2  import build_opener, Request, HTTPError
from urlparse import urlparse
from httplib import BadStatusLine

try:
    from Utilities.RelMon.authentication import X509CertOpen
except ImportError:
    from authentication import X509CertOpen

##-----------------   Make files pairs:  RelValData utils   --------------------
def get_relvaldata_id(file):
    """Returns unique relvaldata ID for a given file."""
    run_id = re.search('R\d{9}', file)
    run = re.search('_RelVal_([\w\d]*)-v\d__', file)
    if not run:
        run = re.search('GR_R_\d*_V\d*C?_([\w\d]*)-v\d__', file)
    if run_id and run:
        return (run_id.group(), run.group(1))
    return None

def get_relvaldata_cmssw_version(file):
    """Returns tuple (CMSSW release, GR_R version) for specified RelValData file."""
    cmssw_release = re.findall('(CMSSW_\d*_\d*_\d*(?:_[\w\d]*)?)-', file)
    gr_r_version = re.findall('-(GR_R_\d*_V\d*\w?)(?:_RelVal)?_', file)
    if not gr_r_version:
        gr_r_version = re.findall('CMSSW_\d*_\d*_\d*(?:_[\w\d]*)?-(\w*)_RelVal_', file)
    if cmssw_release and gr_r_version:
        return (cmssw_release[0], gr_r_version[0])

def get_relvaldata_version(file):
    """Returns tuple (CMSSW version, run version) for specified file."""
    cmssw_version = re.findall('DQM_V(\d*)_', file)
    run_version = re.findall('_RelVal_[\w\d]*-v(\d)__', file)
    if not run_version:
        run_version = re.findall('GR_R_\d*_V\d*C?_[\w\d]*-v(\d)__', file)
    if cmssw_version and run_version:
        return (int(cmssw_version[0]), int(run_version[0]))

def get_relvaldata_max_version(files):
    """Returns file with maximum version at a) beggining of the file,
    e.g. DQM_V000M b) at the end of run, e.g. _run2012-vM. M has to be max."""
    max_file = files[0]
    max_v = get_relvaldata_version(files[0])
    for file in files:
        file_v = get_relvaldata_version(file)
        if file_v[1] > max_v[1] or ((file_v[1] == max_v[1]) and (file_v[0] > max_v[0])):
            max_file = file
            max_v = file_v
    return max_file

## -------------------   Make files pairs:  RelVal utils   ---------------------
def get_relval_version(file):
    """Returns tuple (CMSSW version, run version) for specified file."""
    cmssw_version = re.findall('DQM_V(\d*)_', file)
    run_version = re.findall('CMSSW_\d*_\d*_\d*(?:_[\w\d]*)?-[\w\d]*_V\d*\w?(?:_[\w\d]*)?-v(\d*)__', file)
    if cmssw_version and run_version:
        return (int(cmssw_version[0]), int(run_version[0]))

def get_relval_max_version(files):
    """Returns file with maximum version at a) beggining of the file,
    e.g. DQM_V000M b) at the end of run, e.g. _run2012-vM. M has to be max."""
    max_file = files[0]
    max_v = get_relval_version(files[0])
    for file in files:
        file_v = get_relval_version(file)
        if file_v[1] > max_v[1] or ((file_v[1] == max_v[1]) and (file_v[0] > max_v[0])):
            max_file = file
            max_v = file_v
    return max_file

def get_relval_cmssw_version(file):
    cmssw_release = re.findall('(CMSSW_\d*_\d*_\d*(?:_[\w\d]*)?)-', file)
    gr_r_version = re.findall('CMSSW_\d*_\d*_\d*(?:_[\w\d]*)?-([\w\d]*)_V\d*\w?(_[\w\d]*)?-v', file)
    if cmssw_release and gr_r_version:
        return (cmssw_release[0], gr_r_version[0])

def get_relval_id(file):
    """Returns unique relval ID (dataset name) for a given file."""
    dataset_name = re.findall('R\d{9}__([\w\d]*)__CMSSW_', file)
    return dataset_name[0]

## -----------------------  Make file pairs --------------------------
def is_relvaldata(files):
    is_relvaldata_re = re.compile('_RelVal_')
    return any([is_relvaldata_re.search(filename) for filename in files])

def make_file_pairs(files1, files2):
    print '\n#################       Analyzing files       ###################'
    ## Select functions to use
    if is_relvaldata(files1):
        is_relval_data = True
        get_cmssw_version = get_relvaldata_cmssw_version
        get_id = get_relvaldata_id
        get_max_version = get_relvaldata_max_version
    else:
        is_relval_data = False
        get_cmssw_version = get_relval_cmssw_version
        get_id = get_relval_id
        get_max_version = get_relval_max_version

    ## Divide files into groups
    versions1, versions2 = dict(), dict() # {version1: [file1, file2, ...], version2: [...], ...}
    for files, versions in (files1, versions1), (files2, versions2):
        for file in files:
            version = get_cmssw_version(file)
            if version:
                if version in versions:
                    versions[version].append(file)
                else:
                    versions[version] = [file]

    ## Print the division into groups
    print 'For RELEASE1 found file groups:'
    for version in versions1:
        print '   %s: %d files' % (str(version),  len(versions1[version]))
    if not versions1:
        print 'None.'

    print '\nFor RELEASE2 found file groups:'
    for version in versions2:
        print '   %s: %d files' % (str(version),  len(versions2[version]))
    if not versions2:
        print 'None.'

    if not len(versions1) or not len(versions2):
        print '\nNot enough file groups. Exiting...\n'
        exit()

    ## Pair till you find pairs.
    pairs = []
    for v1 in sorted(versions1, key=lambda x: len(versions1[x]), reverse=True):
        for v2 in sorted(versions2, key=lambda x: len(versions2[x]), reverse=True):
            if v1 == v2:
                continue
            ## Print the groups.
            print '\n#################     Pairing the files     ###################'
            print '%s (%d files)   VS   %s (%d files):\n' % (str(v1),
                    len(versions1[v1]), str(v2), len(versions2[v2]))

            ## Pairing two versions
            for unique_id in set([get_id(file) for file in versions1[v1]]):
                if is_relval_data:
                    dataset_re = re.compile(unique_id[0] + '_')
                    run_re = re.compile(unique_id[1])
                    c1_files = [file for file in versions1[v1] if dataset_re.search(file) and run_re.search(file)]
                    c2_files = [file for file in versions2[v2] if dataset_re.search(file) and run_re.search(file)]
                else:
                    dataset_re = re.compile(unique_id + '_')
                    c1_files = [file for file in versions1[v1] if dataset_re.search(file)]
                    c2_files = [file for file in versions2[v2] if dataset_re.search(file)]

                if len(c1_files) > 0 and len(c2_files) > 0:
                    first_file = get_max_version(c1_files)
                    second_file = get_max_version(c2_files)
                    print '%s\n%s\n' % (first_file, second_file)
                    pairs.append((first_file, second_file))

            print "Got %d pairs." % (len(pairs))
            if pairs:
                return pairs
    print 'Found no file pairs. Exiting..\n'
    exit()

## --------------------   Recursife file downloader -----------------------
def auth_wget(url):
    try:
        opener = build_opener(X509CertOpen())
        return opener.open(Request(url)).read()
    except HTTPError as e:
        print '\nError: DQM GUI is temporarily unavailable. Probably maintainance hours. '+\
                'Please try again later. Original error message: ``%s``. \nExiting...\n' % (e,)
        exit()
    except BadStatusLine as e:
        print '\nYou do not have permissions to access DQM GUI. Please check if your certificates '+\
            'in ``~/.globus`` directory are configured correctly. Exitting...' 
        exit()


def auth_download_file(url, chunk_size=1048576):
    filename = basename(url)
    file_path = join(auth_download_file.work_dir, filename)

    file = open(file_path, 'wb')
    opener = build_opener(X509CertOpen())
    url_file = opener.open(Request(url))
    chunk = url_file.read(chunk_size)
    while chunk:
        file.write(chunk)
        auth_download_file.q.put((1,))   # reports, that downloaded 1MB
        chunk = url_file.read(chunk_size)
    print '\rDownloaded: %s  ' % (filename,)
    file.close()


def recursive_search_online(url, rel1, frags1, rel2, frags2):
    """Recursively searches for files, that matches the pattern."""
    if not url:
        url = 'https://cmsweb.cern.ch/dqm/relval/data/browse/ROOT/RelValData/'
        g1, g2 = recursive_search_online(url, rel1, frags1, rel2, frags2)
        url = 'https://cmsweb.cern.ch/dqm/relval/data/browse/ROOT/RelVal/'
        g3, g4 = recursive_search_online(url, rel1, frags1, rel2, frags2)
        g1.update(g3), g2.update(g4)
        return g1, g2

    domain = '://'.join(urlparse(url)[:2])

    ## Compile regular expressions
    href_re = re.compile(r"<a href='([-./\w]*)'>([-./\w]*)<")

    def compile_res(rel, frags):
        frags = frags.split(',')
        regexps = [s for s in frags if not s.startswith('!')]
        regexps += ['^((?%s).)*$' % s for s in frags if s.startswith('!')]
        regexps += [rel + '-', '.root']
        return [re.compile(r) for r in regexps]

    res1 = compile_res(rel1, frags1)
    res2 = compile_res(rel2, frags2)

    ## Recursively find files that matches regular expressions
    hrefs = [(name, path) for path, name in href_re.findall(auth_wget(url))[1:]]
    files_with_urls1, files_with_urls2 = dict(), dict()
    for name, path in hrefs:
        if splitext(name)[1]: # If file
            if all([r.search(name) for r in res1]):
                files_with_urls1[name] = domain + path
            if all([r.search(name) for r in res2]):
                files_with_urls2[name] = domain + path
        else:
            print domain + path
            new_hrefs = href_re.findall(auth_wget(domain + path))[1:]
            hrefs.extend([(name, path) for path, name in new_hrefs])
    return files_with_urls1, files_with_urls2

def search_on_disk(work_path, rel1, frags1, rel2, frags2):
    if not work_path:
        print 'No working directory specified. Use "--dir DIR" option to ' +\
              'specify working directory. Exiting...'
        exit()
    ## Compile regular expressions
    def compile_res(rel, frags):
        frags = frags.split(',')
        regexps = [s for s in frags if not s.startswith('!')]
        regexps += ['^((?%s).)*$' % s for s in frags if s.startswith('!')]
        regexps += [rel + '-', '.root']
        return [re.compile(r) for r in regexps]

    res1 = compile_res(rel1, frags1)
    res2 = compile_res(rel2, frags2)

    ## Recursively find files that matches regular expressions
    files = listdir(work_path)
    files1, files2 = [], []
    for name in files:
        if splitext(name)[1]:
            if all([r.search(name) for r in res1]):
                files1.append(name)
            if all([r.search(name) for r in res2]):
                files2.append(name)
    return files1, files2


## Exception definitions
comparison_errors = {
        'Missing histogram': -1,
        'Histograms have different types': -2,
        'Object is not a histogram': -3,
        'Ranges of histograms are different': -4
    }

class ComparisonError(Exception):
    def __init__(self, error_message, *args, **kwargs):
        self.error_message = error_message
        self.error_code = comparison_errors[error_message]

    def __str__(self):
        return 'Comparison Error: %d' % self.error_code


## StatisticalTests
class StatisticalTest(object):
    name = None

    def get_N_bins(self, h):
        x = h.GetNbinsX()
        y = h.GetNbinsY()
        z = h.GetNbinsZ()
        if not (y and z): # Is this realy necessary?
            return 0
        return (x + 1) * (y + 1) * (z + 1)

    def is_empty(self, h):
        for i in xrange(1, self.get_N_bins(h)):
            if h.GetBinContent(i) != 0:
                return False
            return True

    def do_test(self, h1, h2):
        if not h1 or not h2:
            raise ComparisonError('Missing histogram')
        if not isinstance(h1, type(h2)):
            return -104     # raise ComparisonError('Histograms have different types')
        if not h1.InheritsFrom('TH1'):
            return -105     # raise ComparisonError('Object is not a histogram')
        if self.is_empty(h1) or self.is_empty(h2):
            return 1
        h1_bins = self.get_N_bins(h1)
        if h1_bins != self.get_N_bins(h2):
            return -103     # raise CoparisonError('Ranges of histograms are different')


class KolmogorovTest(StatisticalTest):
    name = 'KS'

    def do_test(self, h1, h2):
        p_value = super(KolmogorovTest, self).do_test(h1, h2)
        if p_value is not None:
            return p_value

        for h in h1, h2:
            if h.GetSumw2().GetSize() == 0:
                h.Sumw2()
        return h1.KolmogorovTest(h2)


class Chi2Test(StatisticalTest):
    name = 'Chi2'

    def make_absolute(self, h, bin_count):
        for i in xrange(1, bin_count): # Why here is no +1?
            content = h.GetBinContent(i)
            if content < 0:
                h.SetBinContent(i, -1 * content)
            if h.GetBinError(i) == 0 and content != 0:
                h.SetBinContent(i, 0)

    def enough_filled_bins(self, h, bin_count, more_than=3):
        filled_bins = 0
        for i in xrange(1, bin_count):
            if h.GetBinContent(i) > 0:
                filled_bins += 1
            if filled_bins > more_than:
                return True
        return False

    def do_test(self, h1, h2):
        p_value = super(Chi2Test, self).do_test(h1, h2)
        if p_value is not None:
            return p_value

        bin_count = self.get_N_bins(h1)

        # Make histograms absolute.
        self.make_absolute(h1, bin_count)
        self.make_absolute(h2, bin_count)

        # Check if there is enough filled bins in bouth histograms.
        if not self.enough_filled_bins(h1, bin_count) or\
           not self.enough_filled_bins(h2, bin_count):
            return 1

        if h1.InheritsFrom("TProfile") or (h1.GetEntries() != h1.GetSumOfWeights()):
            return h1.Chi2Test(h2, 'WW')
        return h1.Chi2Test(h2, 'UU')


tests = {KolmogorovTest.name: KolmogorovTest, Chi2Test.name: Chi2Test}

## Utils
def init_database(db_path):
    print 'Initialising DB: %s...' % basename(db_path),
    conn = sqlite3.connect(db_path)

    ## Creates tables
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS ReleaseComparison (
                        id INTEGER PRIMARY KEY,
                        title TEXT,
                        release1 TEXT,
                        release2 TEXT,
                        statistical_test TEXT
                    );""")
    c.execute("""CREATE TABLE IF NOT EXISTS Directory (
                        id INTEGER PRIMARY KEY,
                        name TEXT,
                        parent_id INTEGER,
                        from_histogram_id INTEGER,
                        till_histogram_id INTEGER,
                        FOREIGN KEY (parent_id) REFERENCES Directory(id)
                        FOREIGN KEY (from_histogram_id) REFERENCES HistogramComparison(id)
                        FOREIGN KEY (till_histogram_id) REFERENCES HistogramComparison(id)
                    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS RootFileComparison (
                        id INTEGER PRIMARY KEY,
                        filename1 TEXT,
                        filename2 TEXT,
                        release_comparison_id INTEGER,
                        directory_id INTEGER,
                        FOREIGN KEY (release_comparison_id) REFERENCES ReleaseComparison(id),
                        FOREIGN KEY (directory_id) REFERENCES Directory(id)
                    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS HistogramComparison (
                        id INTEGER PRIMARY KEY,
                        name TEXT,
                        p_value REAL,
                        directory_id INTEGER,
                        FOREIGN KEY (directory_id) REFERENCES Directory(id)
                    )""")

    print 'Done.'
    return db_path


def get_version(filename):
    """Returns CMSSW and GR_R versions for the given filename."""
    if is_relvaldata([filename]):
        version_elems = get_relvaldata_cmssw_version(filename)
    else:
        relval_version = get_relval_cmssw_version(filename)
        version_elems = (relval_version[0], relval_version[1][0], relval_version[1][1])
    version_elems = [elem.strip('_').strip('RelVal_') for elem in version_elems]
    return '___'.join([elem for elem in version_elems if elem])


def get_size_to_download(work_path, files_with_urls):
    """Returns file list to download and total size to download."""
    opener = build_opener(X509CertOpen())
    size_to_download = 0
    files_to_download = []
    for filename, url in files_with_urls:
        url_file = opener.open(Request(url))
        size = int(url_file.headers["Content-Length"])
        file_path = join(work_path, filename)
        if exists(file_path) and getsize(file_path) / 1024 == size / 1024:
            print "Exists on disk %s." % filename
        else:
            size_to_download += size
            files_to_download.append(url)
    return size_to_download, files_to_download

def check_disk_for_space(work_path, size_needed):
    '''Checks afs file system for space.'''
    pass
    # try:
    #     fs_proc = subprocess.Popen(['fs', 'listquota', work_path], stdout=subprocess.PIPE)
    # except OSError:
    #     return
    # fs_response = fs_proc.communicate()[0]
    # quota, used = re.findall('([\d]+)', fs_response)[:2]
    # free_space = int(quota) - int(used)
    # if free_space * 1024 < size_needed:
    #     print '\nNot enougth free space on disk.',
    #     print 'Free space: %d MB. Need: %d MB. Exiting...\n' % (free_space / 1024, size_needed /1048576)
    #     exit()
    # elif size_needed:
    #     print 'Free space on disk: %d MB.\n' % (free_space / 1024,)


def show_status_bar(total_size):
    """Shows download status."""
    q = show_status_bar.q
    total_size = total_size / (1024*1024)
    downloaded = 0
    while downloaded < total_size:
        try:
            o = q.get(timeout=20)
            downloaded += 1
            print '\r      %d/%d MB     %d%%     ' % (downloaded, total_size, 100*downloaded/total_size),
            sys.stdout.flush()
        except Empty:
            time.sleep(1)
            break

#! /usr/bin/env python
# Compute the sample size from das and in location 
# Victor E. Bazterra UIC (2012)
#

import exceptions, json, os, subprocess, tempfile, time
from optparse import OptionParser

def DASQuery(query, ntries = 6, sleep = 1):
    """
    Query das expecting to execute python client
    """
    das = '%s/src/TopQuarkAnalysis/TopPairBSM/test/das.py' % os.environ['CMSSW_BASE']
    dasquery = '%s --query "%s" --format json' % (das, query)
    for n in range(ntries):
        stdout = tempfile.TemporaryFile()
        subprocess.check_call(dasquery, shell = True, stdout=stdout)
        stdout.seek(0)
        payload = json.load(stdout)
        if not payload['nresults'] > 0:
            time.sleep(sleep)
            sleep = 2*sleep
            continue
        return payload
    print 'Warning failed das quary after %d iterations' % ntries
    return None


def GetDCacheLocation(name):
    """
    Get the dirname of the first file of the dataset
    """
    dasquery = 'file dataset = %s instance = cms_dbs_ph_analysis_02 | grep file.name' % name
    payload = DASQuery(dasquery)
    if not payload: return ''
    location = ''
    try:
        location = os.path.dirname(payload['data'][0]['file'][0]['name'])
    except:
        print 'Warning error when parsing the dataset location.'
        location = ''
    return location


def GetDatasetSizeInDas(name):
    """
    Get dataset size from das
    """
    dasquery = 'dataset dataset = %s instance = cms_dbs_ph_analysis_02 | grep dataset.size' % name
    payload = DASQuery(dasquery)
    if not payload: return 0
    size = 0
    try:
        size = float(payload['data'][0]['dataset'][0]['size'])
    except:
        print 'Warning error when parsing the dataset size'
        size = 0
    return size


def GetDatasetSizeInLocation(name):
    """
    Get dataset size from das
    """
    cmd = 'du -b %s' % name
    stdout = tempfile.TemporaryFile()
    subprocess.check_call(cmd, shell = True, stdout=stdout)
    stdout.seek(0)
    return float(stdout.read().split()[0])


def SizeFormat(uinput):
    """
    Format file size utility, it converts file size into KB, MB, GB, TB, PB units
    """
    try:
        num = float(uinput)
    except Exception as _exc:
        return uinput
    base = 1000. # power of 10, or use 1024. for power of 2
    for xxx in ['', 'KB', 'MB', 'GB', 'TB', 'PB']:
        if  num < base:
            return "%3.1f %s" % (num, xxx)
        num /= base


def GetDatasetSize(name, whitelist=['lpctlbsm','b2g12006']):
    """
    Delete dataset main function
    """
    prefix = '/pnfs/cms/WAX/11'
    print 'Processing dataset %s' % name
    location = GetDCacheLocation(name)
    print '  location %s' % location
    found = False
    if location != '':
        for dir in whitelist:
            if dir in location:
                found = os.path.exists(prefix+location)
                break
    print '  found in location', found
    locsize = 0
    dassize = 0
    if location != '': dassize = GetDatasetSizeInDas(name)
    print '  das size %s' % SizeFormat(dassize)
    if found:
        locsize = GetDatasetSizeInLocation(prefix+location)
        print '  loc size %s' % SizeFormat(locsize)
    print
    return dassize, locsize


def main():

    if not 'CMSSW_BASE' in os.environ:
        raise exceptions.RuntimeError('CMSSW_BASE is not setup.')

    usage = 'usage: %prog getouput [options]\n'
    usage = usage + 'Compute the sample size from das and in location.'

    parser = OptionParser(usage = usage)

    parser.add_option(
        '--filelist', type='string',
        help='List of dataset to be deleted.'
    )

    (options, args) = parser.parse_args()

    if os.path.isfile(options.filelist):
        tdassize = 0 
        tlocsize = 0
        filelist = open(options.filelist)
        for file in filelist.readlines():
            dassize, locsize = GetDatasetSize(file.rstrip())
            tdassize = tdassize + dassize
            tlocsize = tlocsize + locsize
        print 'Total size %s (das), %s (loc)' % (SizeFormat(tdassize), SizeFormat(tlocsize)) 
    else:
        raise exceptions.IOError('Filelist %s does not exist.' % options.filelist)

if __name__ == "__main__":
    main()


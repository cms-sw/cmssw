#! /usr/bin/env python
# Create initial production table 
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


def GetDatasetSites(name):
    """
    Get dataset number of events from das
    """
    dasquery = 'site dataset = %s' % name
    payload = DASQuery(dasquery)
    sites = {}
    try:
        for site in payload['data']:
            site = site['site'][0]
            sites[site['name']] = float(site['dataset_fraction'].rstrip('%'))
    except:
        raise exceptions.ValueError('Error when parsing the dataset sites')
    return sites


def SelectedSites(dataset):
    """
    Return the a bool if the selected sites contain the dataset (FNAL or any T2_US).
    """
    print 'Looking for sites to process AOD: %s' % dataset
    sites = GetDatasetSites(dataset)
    result = False
    for site, frac in sites.iteritems():
        if 'FNAL' in site or 'T2_US' in site and frac == 100.0:
            result = True
        print 'Site %s (%0.1f)'% (site,frac)
    return result


def DataProductionTable(datasets):
    print '%TABLE{"headerrows="1"}%'
    print '%EDITTABLE{ format="| label | select, 1, OUTSITE, INSITE, SUBMITTED, DONE, ELEVATING, ELEVATED, TRANSFERRED, DELETED | text, 40 | text, 8 | text, 100 |" changerows="off" }%'
    print '| *Parent Sample* | *Status* | *Submitter* | *Lumi [pb-1]* | *PAT* |'
    for dataset in sorted(datasets.keys()):
        if datasets[dataset]:
            print '| =%s= | INSITE | full name | 0 | =PAT= |' % dataset
        else:
            print '| =%s= | OUTSITE | full name | 0 | =PAT= |' % dataset


def MCProductionTable(datasets):
    print '%TABLE{"headerrows="1"}%'
    print '%EDITTABLE{ format="| label | select, 1, OUTSITE, INSITE, SUBMITTED, DONE, ELEVATING, ELEVATED, TRANSFERRED, DELETED | text, 40 | text, 8 | text, 8 | text, 100 |" changerows="off" }%'  
    print '| *Parent Sample* | *Status* | *Submitter* | *N_total* | *N_selected* | *PAT* |'
    for dataset in sorted(datasets.keys()):
        if datasets[dataset]:
            print '| =%s= | INSITE | full name | 0 | 0 | =PAT= |' % dataset
        else:
            print '| =%s= | OUTSITE | full name | 0 | 0 | =PAT= |' % dataset


def main():

    if not 'CMSSW_BASE' in os.environ:
        raise exceptions.RuntimeError('CMSSW_BASE is not setup.')

    usage = 'usage: %prog [options]\n'
    usage = usage + 'Create initial production table.'

    parser = OptionParser(usage = usage)

    parser.add_option(
        '--filelist', type='string',
        help='List of dataset to be deleted.'
    )

    parser.add_option(
        '--type', type='string', default='mc',
        help='Type of production table (data or mc, default mc).'
    )

    (options, args) = parser.parse_args()

    if os.path.isfile(options.filelist):
        filelist = open(options.filelist)
        datasets = {}
        for dataset in sorted(filelist.readlines()):
            if SelectedSites(dataset.rstrip()):
                print 'Dataset in selected sites (INSITE).\n'
                datasets[dataset.rstrip()] = True
            else:
                print
                datasets[dataset.rstrip()] = False
        if options.type.lower() == 'mc':
            MCProductionTable(datasets) 
        elif options.type.lower() == 'data':
            DataProductionTable(datasets)
        else:
            raise exceptions.ValueError('Error type %s indefined (option are data or mc).' % options.type.lower()) 
    else:
        raise exceptions.IOError('Filelist %s does not exist.' % options.filelist)

if __name__ == "__main__":
    main()


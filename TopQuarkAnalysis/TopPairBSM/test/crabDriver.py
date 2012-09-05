#! /usr/bin/env python
# Compute the sample size from das and in location 
# Victor E. Bazterra UIC (2012)
#

import exceptions, json, os, string, subprocess, tempfile, time
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
    raise exceptions.StopIteration('Failed das quary after %d iterations' % ntries)


def GetDatasetSites(name):
    """
    Get dataset number of events from das
    """
    dasquery = 'site dataset = %s' % name
    payload = DASQuery(dasquery)
    sites = []
    try:
        for site in payload['data']:
            site = site['site'][0]
            dataset_fraction = float(site['dataset_fraction'].rstrip('%'))
            if dataset_fraction == 100.0:
                sites.append(site['name'])
    except:
        raise exceptions.ValueError('Error when parsing the dataset sites')
    return sites


def SelectSite(dataset):
    """
    Select the site to run (FNAL or T2_US)
    """
    sites = GetDatasetSites(dataset)
    result = ''
    for site in sites:
        if 'FNAL' in site:
            result = 'FNAL'
            break
        elif 'T2_US' in site:
            result = 'T2_US'
    return result


def GetDatasetNEvents(name):
    """
    Get dataset number of events from das
    """
    dasquery = 'dataset dataset = %s | grep dataset.nevents' % name
    payload = DASQuery(dasquery)
    nevents = 0
    try:
        nevents = float(payload['data'][0]['dataset'][0]['nevents'])
    except:
        raise exceptions.ValueError('Error when parsing the dataset size')
    return nevents


def CreateDataCrabConfig(options):
    """
    Creates a crab config file for processing data
    """
    if not options.lumimask:
        raise exceptions.ValueError('Missing lumimask for a data job.')
    elif not os.path.isfile(options.lumimask):
        raise exceptions.IOError('Lumimask file %s does not exist.' % options.lumimask)

    scheduler = 'condor'
    use_server = '0'
    grid = ''

    site = SelectSite(options.dataset)

    if site == 'T2_US':
        scheduler = 'glidein'
        use_server = '1'
        grid = '[GRID]\nse_white_list = T2_US_*'  
    elif site != 'FNAL':
        raise exceptions.ValueError('Neither FNAL nor T2_US have the dataset.')

    nevents = GetDatasetNEvents(options.dataset)    
    njobs = int(nevents*options.eventsize/(1000*options.filesize))

    datasetblock = 'total_number_of_lumis = -1\n'
    datasetblock = datasetblock + 'number_of_jobs = %d\n' % int(njobs)
    datasetblock = datasetblock + 'lumi_mask = %s' % options.lumimask
   
    pycfg_params = 'tlbsmTag=%s useData=1' % options.tag.lower()
    if options.pycfg :
        pycfg_params = pycfg_params + ' ' + options.pycfg

    publish_data_name = options.dataset.split('/')[2] + '_' + options.tag
    ui_working_dir = options.dataset.replace('/AOD','').replace('/','_')[1:] + '_' + options.tag

    settings = {
        'scheduler': scheduler,
        'use_server': use_server,
        'datasetpath': options.dataset,
        'pycfg_params': pycfg_params,
        'datasetblock': datasetblock,
        'publish_data_name': publish_data_name,
        'ui_working_dir': ui_working_dir,
        'grid': grid
    }

    filename = '%s/src/TopQuarkAnalysis/TopPairBSM/test/crab_template.cfg' % os.environ['CMSSW_BASE']
    file = open(filename)
    template = string.Template(file.read())
    file.close()
    file = open('crab_%s.cfg' % ui_working_dir, 'w')
    file.write(template.safe_substitute(settings))
    file.close()


def CreateMCCrabConfig(options):
    """
    Creates a crab config file for processing mc 
    """

    scheduler = 'condor'
    use_server = '0'
    grid = ''

    site = SelectSite(options.dataset)

    if site == 'T2_US':
        scheduler = 'glidein'
        use_server = '1'
        grid = '[GRID]\nse_white_list = T2_US_*'
    elif site != 'FNAL':
        raise exceptions.ValueError('Neither FNAL nor T2_US have the dataset.')

    nevents = GetDatasetNEvents(options.dataset)    
    njobs = int(nevents*options.eventsize/(1000*options.filesize))

    datasetblock = 'total_number_of_events = -1\n'
    datasetblock = datasetblock + 'number_of_jobs = %d' % int(njobs)
   
    pycfg_params = 'tlbsmTag=%s useData=0' % options.tag.lower()
    if options.pycfg :
        pycfg_params = pycfg_params + ' ' + options.pycfg

    publish_data_name = options.dataset.split('/')[2] + '_' + options.tag
    ui_working_dir = options.dataset.replace('/AODSIM','').replace('/','_')[1:] + '_' + options.tag

    settings = {
        'scheduler': scheduler,
        'use_server': use_server,
        'datasetpath': options.dataset,
        'pycfg_params': pycfg_params,
        'datasetblock': datasetblock,
        'publish_data_name': publish_data_name,
        'ui_working_dir': ui_working_dir,
        'grid': grid
    }

    filename = '%s/src/TopQuarkAnalysis/TopPairBSM/test/crab_template.cfg' % os.environ['CMSSW_BASE']
    file = open(filename)
    template = string.Template(file.read())
    file.close()
    file = open('crab_%s.cfg' % ui_working_dir, 'w')
    file.write(template.safe_substitute(settings))
    file.close()


def main():

    if not 'CMSSW_BASE' in os.environ:
        raise exceptions.RuntimeError('CMSSW_BASE is not setup.')

    usage = 'usage: %prog getouput [options]\n'
    usage = usage + 'Delete from dcache a all the files for a given dataset.'

    parser = OptionParser(usage = usage)

    parser.add_option(
        '--dataset', type='string',
        help='Dataset in AOD or AODSIM format use as input to tlbsm.'
    )

    parser.add_option(
        '--lumimask', type='string',
        help='Name of the json file use as lumimask.'
    )

    parser.add_option(
        '--tag', type='string', default='TLBSM_53x_v2',
        help='Version of tlbsm, the tag has to follow the format TLBSM_XXX_vX.'
    )

    parser.add_option(
        '--pycfg', type='string',
        help='Version of tlbsm, the tag has to follow the format TLBSM_XXX_vX.'
    )

    parser.add_option(
        '--eventsize', type='float', default=65.0,
        help='Event size in Kb use to compute the number of crab jobs.'
    )

    parser.add_option(
        '--filesize', type='float', default=500,
        help='File size in Mb produced per job use to compute the number of crab jobs.'
    )

    (options, args) = parser.parse_args()

    if not 'AODSIM' in options.dataset:
        CreateDataCrabConfig(options)
    elif 'AOD' in options.dataset:
        CreateMCCrabConfig(options)
    else:
        raise exceptions.ValueError('Dataset privided is not AOD or AODSIM.')

if __name__ == "__main__":
    main()


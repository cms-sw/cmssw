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
    dasquery = '%s --query "%s" --format json --limit 10000' % (das, query)
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


def GetDatasetSites(name, fraction = 100.0):
    """
    Get dataset number of events from das
    """
    dasquery = 'site dataset = %s' % name
    payload = DASQuery(dasquery)
    sites = {}
    try:
        for site in payload['data']:
            site = site['site'][0]
            dataset_fraction = 0.0
            if 'dataset_fraction' in site: 
                dataset_fraction = float(site['dataset_fraction'].rstrip('%'))
            elif 'replica_fraction' in site:
                dataset_fraction = float(site['replica_fraction'].rstrip('%')) 
            if dataset_fraction >= fraction:
                sites[site['name']] = dataset_fraction 
    except:
        raise exceptions.ValueError('Error when parsing the dataset sites')
    return sites


def SelectSite(dataset, fraction=100.0):
    """
    Select the site to run (FNAL or T2_US)
    """
    sites = GetDatasetSites(dataset, fraction)
    result = ''
    for site in sites.keys():
        if 'T2_US' in site:
            result = 'T2_US'
            break
        elif 'FNAL' in site:
            result = 'FNAL'
        elif 'T2' in site and result != 'FNAL':
            result = 'T2'
    report = ''
    for site in sites.keys():
        report = report + ', %s(%%%0.2f)' % (site, sites[site]) 
    print 'Dataset located at : %s' % report[2:]
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


def CreateCrabConfig(options, isdata):
    """
    Creates a crab config file for processing data
    """
    if isdata:
        if not options.lumimask:
            raise exceptions.ValueError('Missing lumimask for a data job.')
        elif not os.path.isfile(options.lumimask):
            raise exceptions.IOError('Lumimask file %s does not exist.' % options.lumimask)

    scheduler = 'condor'
    use_server = '0'
    grid = ''

    nevents = GetDatasetNEvents(options.dataset)
    njobs = int(nevents*options.eventsize/(1000*options.filesize))

    site = SelectSite(options.dataset, options.fraction)

    if site == 'T2_US':
        scheduler = 'remoteGlidein'
        use_server = '0'
        grid = '[GRID]\nse_white_list = T2_US_*'
    elif site == 'T2':
        print 'Warning: Neither FNAL nor T2_US have the dataset.'
        print 'This could mean more chances of problems in the stageout (exit code 60317).'
        print 'Increasing the number of jobs by facto 4.'
        scheduler = 'remoteGlidein'
        use_server = '0'
        njobs = 4*njobs
    elif site != 'FNAL':
        raise exceptions.ValueError('No T2 site contains this dataset.')

    if njobs > 5000:
        print 'Warning: the number of jobs for this samples was reduce to 5000.'
        njobs = 5000

    if not isdata:
        datasetblock = 'total_number_of_events = -1\n'
        datasetblock = datasetblock + 'number_of_jobs = %d' % int(njobs)
    
        pycfg_params = 'tlbsmTag=%s useData=0' % options.tag.lower()
        if options.pycfg : 
            pycfg_params = pycfg_params + ' ' + options.pycfg
        
        publish_data_name = options.dataset.split('/')[2] + '_' + options.tag
        ui_working_dir = options.dataset.replace('/AODSIM','').replace('/','_')[1:] + '_' + options.tag
    else:
        datasetblock = 'total_number_of_lumis = -1\n'
        datasetblock = datasetblock + 'number_of_jobs = %d\n' % int(njobs) 
        datasetblock = datasetblock + 'lumi_mask = %s' % options.lumimask
        
        pycfg_params = 'tlbsmTag=%s useData=1' % options.tag.lower()
        if options.pycfg :  
            pycfg_params = pycfg_params + ' ' + options.pycfg
    
        publish_data_name = options.dataset.split('/')[2] + '_' + options.tag
        ui_working_dir = options.dataset.replace('/AOD','').replace('/','_')[1:] + '_' + options.tag

    if options.extension > 0 and isdata:
        publish_data_name = publish_data_name + '_extension_v%d' % options.extension
        ui_working_dir = ui_working_dir + '_extension_v%d' % options.extension

    if options.bugfix > 0:
        publish_data_name = publish_data_name + '_bugfix_v%d' % options.bugfix
        ui_working_dir = ui_working_dir + '_bugfix_v%d' % options.bugfix

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
        '--fraction', type='float', default=100.0,
        help='Minimal fraction tolerated of the dataset present in the sites.'
    )

    parser.add_option(
        '--lumimask', type='string',
        help='Name of the json file use as lumimask.'
    )

    parser.add_option(
        '--tag', type='string', default='TLBSM_53x_v3',
        help='Version of tlbsm, the tag has to follow the format TLBSM_XXX_vX.'
    )

    parser.add_option(
        '--extension', type='int', default=0,
        help='Version when extending a dataset (starting from 1).'
    )

    parser.add_option(
        '--bugfix', type='int', default=0,
        help='Version of bugfix when redoing a dataset (starting from 1).'
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
        CreateCrabConfig(options, isdata = True)
    elif 'AOD' in options.dataset:
        CreateCrabConfig(options, isdata = False)
    else:
        raise exceptions.ValueError('Dataset privided is not AOD or AODSIM.')

if __name__ == "__main__":
    main()


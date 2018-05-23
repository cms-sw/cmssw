#!/usr/bin/env python
# coding: utf-8
'''
Helper functions for CherryPy application ``browse_db.py``.

Author:  Albertas Gimbutas,  Vilnius University (LT)
e-mail:  albertasgim@gmail.com
'''

import sqlite3
import re
from os import getcwd, listdir
from os.path import join
from urllib import quote
from functools import reduce


renaming = {
        'MessageLogger': 'Miscellanea', 'FourVector': 'Generic',
        'Castor': 'Castor Calorimeter', 'RPCDigisV': 'Resistive Plate Chambers',
        'GlobalRecHitsV': 'Miscellanea: Sim.', 'Top': 'Top', 'HLTJETMET': 'JetMet',
        'GlobalDigisV': 'Miscellanea: Sim.', 'L1TEMU': 'Level 1 Trigger',
        'TrackerRecHitsV': 'Tracking System', 'MuonDTHitsV': 'Muon Objects',
        'EcalDigisV': 'Ecal Calorimeter', 'EcalHitsV': 'Ecal Calorimeter',
        'Muons': 'Muon Objects', 'DT': 'Drift Tubes', 'TrackerDigisV': 'Tracking System',
        'Pixel': 'Tracking System', 'EcalPreshower': 'Ecal Calorimeter',
        'EgammaV': 'Photons', 'AlCaEcalPi0': 'Alca', 'SusyExo': 'SusyExo',
        'MuonDTDigisV': 'Muon Objects', 'TauRelVal': 'Tau',
        'HcalHitsV': 'Hcal Calorimeter', 'RPC': 'Resistive Plate Chambers',
        'EcalRecHitsV': 'Ecal Calorimeter', 'EgOffline': 'EGamma',
        'MuonCSCDigisV': 'Muon Objects', 'ParticleFlow': 'Miscellanea',
        'Info': 'Miscellanea', 'Tracking': 'Tracking',
        'NoiseRatesV': 'Miscellanea: Sim.', 'Generator': 'Miscellanea: Sim.',
        'Btag': 'B Tagging', 'Higgs': 'Higgs', 'GlobalHitsV': 'Miscellanea: Sim.',
        'HcalRecHitsV': 'Hcal Calorimeter', 'TrackerHitsV': 'Tracking System',
        'CSC': 'Cathode Strip Chambers', 'Muon,HLTMonMuon': 'Muon',
        'Hcal': 'Hcal Calorimeter', 'TauOffline': 'Tau',
        'HeavyFlavor': 'HeavyFlavor', 'JetMET': 'Jet', 'Physics': 'Miscellanea',
        'CaloTowersV': 'Hcal Calorimeter', 'SiStrip': 'Tracking System',
        'EcalClusterV': 'Ecal Calorimeter', 'HLTEgammaValidation': 'EGamma',
        'EcalPhiSym': 'Alca', 'L1T': 'Level 1 Trigger', 'MixingV': 'Miscellanea: Sim.',
        'FourVector_Val': 'Generic', 'EcalEndcap': 'Ecal Calorimeter',
        'TauOnline': 'Tau', 'Egamma': 'Photons', 'HcalIsoTrack': 'Alca',
        'EcalBarrel': 'Ecal Calorimeter'
}


def get_img_path(filename, path):
    '''Returns image path for https://cmsweb.cern.ch/dqm histogram
    visualisation service'''
    run = int(re.findall('_R(\d*)__', filename)[0])
    parts = [e.rstrip('.root') for e in filename.split('__')]
    path = path.replace('Run summary/', '')
    return 'archive/%s/%s/%s/%s/%s' % (run, parts[1], parts[2], parts[3], path)


def get_img_url(path, f1, f2=None, w=250, h=250):
    '''Returns full URL of histogram (or histogram overlay) image for
    https://cmsweb.cern.ch/dqm visualisation service.'''
    base = 'https://cmsweb.cern.ch/dqm/relval/plotfairy'
    if not f2:
        return '%s/%s?w=%s;h=%s' % (base, get_img_path(f1, path), w, h)
    return '%s/overlay?obj=%s;obj=%s;w=%s;h=%s' % (base,
                 get_img_path(f1, path), get_img_path(f2, path), w, h)


def get_dataset_name(name):
    '''Returns extracted dataset name from the given ROOT filename.'''
    if re.search('RelVal', name):
        run = str(int(re.findall('_R(\d{9})_', name)[0]))
        ds = re.findall('GR_R_\d*_V\d*C?_(?:RelVal)?_([\w\d]*-v\d+)_', name)[0]
    else:
        run, ds = re.findall('R(\d{9})__([\w\d]*)__CMSSW_', name)[0:1]
    return '_'.join([ds, str(int(run))])


def get_release(name):
    '''Returns extracted release from the given ROOT filename.'''
    return re.findall('R\d{9}__([\w\d_-]*)__DQM.root', name)[0]


def get_stats(c, threshold, dir_ranges):
    '''Returns ``successes``, ``fails``, ``nulls`` for the given dir_ranges.'''
    successes, nulls, fails = 0, 0, 0
    for from_id, till_id in dir_ranges:
        c.execute('''SELECT count(*) FROM HistogramComparison
                     WHERE p_value >= 0 AND p_value > ? AND
                     id >= ? and id <= ?''', (threshold, from_id, till_id))
        successes += c.fetchone()[0]
        c.execute('''SELECT count(*) FROM HistogramComparison WHERE
                     p_value < 0 AND id >= ? AND id <= ?''', (from_id, till_id))
        nulls += c.fetchone()[0]
        c.execute('''SELECT count(*) FROM HistogramComparison
                     WHERE p_value >= 0 AND p_value <= ? AND
                     id >= ? AND id <= ?''', (threshold, from_id, till_id))
        fails += c.fetchone()[0]
    return successes, nulls, fails


def get_percentage(successes, nulls, fails):
    '''Converts integers ``successes``, ``nulls`` and ``fails`` to percents.'''
    if successes is None:
        return None, None, None
    total = successes + fails + nulls
    if not total:
        return None, None, None
    success =  round(100. * successes / total, 2)
    null =  round(100. * nulls / total, 2)
    fail =  round(100. * fails / total, 2)
    return success, null, fail


def get_folders(c, file_id, filename, dir_id, threshold):  # TODO: If folder [Egamma|JetMet] analyse their subdirs
    '''Returns file folder stats for one "summary table" column.'''
    ds_name = get_dataset_name(filename)
    c.execute('''SELECT name, from_histogram_id, till_histogram_id FROM
                 Directory WHERE parent_id=?''', (dir_id,))
    dirs = c.fetchall()
    file_folders = dict()
    total_successes, total_nulls, total_fails = 0, 0, 0
    for name, from_id, till_id in dirs:
        successes, nulls, fails = get_stats(c, threshold, ((from_id, till_id),))
        total_successes += successes
        total_nulls += nulls
        total_fails += fails
        if name in file_folders:
            file_folders[name].append([file_id, ds_name, successes, nulls, fails])
        else:
            file_folders[name] = [file_id, ds_name, successes, nulls, fails]
    return [('Summary', [file_id, ds_name, total_successes, total_nulls, total_fails])] + file_folders.items()


def join_ranges(ranges, elem):
    '''To do less DB calls, joins [(from_id, till_id), ...] ranges.'''
    if isinstance(ranges, tuple):
        ranges = [ranges]
    if ranges[-1][-1] + 1 == elem[0]:
        ranges[-1] = (ranges[-1][0], elem[1])
    else:
        ranges.append(elem)
    return ranges


def get_release_list(c):
    '''Returns all ``ReleaseComparisons`` found on database.'''
    c.execute('SELECT title, statistical_test FROM ReleaseComparison')
    return c.fetchall()


def db_list_with_releases(path='.'):
    '''Returns available database list and their releases.'''
    db_list = [db for db in listdir(path) if db.endswith('.db')]
    db_list_with_releases = []
    for db in db_list:
        conn = sqlite3.connect(join(path, db))
        releases = get_release_list(conn.cursor())
        db_list_with_releases.append((db[:-3], releases))
        conn.close()
    return db_list_with_releases

# -------------------     Template Context generators     --------------------

def get_release_summary_stats(c, release_title, st_test, threshold=1e-5):
    '''Returns context for ``release_summary.html`` template.'''
    ## Summary
    context = dict()
    c.execute('''SELECT release1, release2, id FROM ReleaseComparison
                 WHERE title = ? AND statistical_test = ?''', (release_title, st_test))
    context['release1'], context['release2'], release_comp_id = c.fetchone()

    # All directory ranges
    c.execute('''SELECT from_histogram_id, till_histogram_id FROM Directory
                   WHERE id IN (SELECT directory_id FROM RootFileComparison
                   WHERE release_comparison_id = ?)''', (release_comp_id,))
    dir_ranges = c.fetchall()

    if len(dir_ranges) > 1:
        dir_ranges = reduce(join_ranges, dir_ranges)

    context['successes'], context['nulls'], context['fails'], = get_stats(c, threshold, dir_ranges)

    context['total'] = context['successes'] + context['fails'] + context['nulls']
    if context['total']:
        context['success'], context['null'], context['fail'] = \
            get_percentage(context['successes'], context['nulls'], context['fails'])

    ## Data needed for the all the statistics:
    c.execute('''SELECT id, filename1, directory_id FROM RootFileComparison
                 WHERE release_comparison_id = ?''', (release_comp_id,))
    files = c.fetchall()

    ## folders: [(folder_name, [folder: (file_id, filename, success, null, fail)]), ...]
    folders = dict()
    for file_id, filename, dir_id in files:
        # file_folders: [(folder_name, [(file_id, file_name, success, null, fail)]), ...]
        file_folders = get_folders(c, file_id, filename, dir_id, threshold)
        for folder_name, file_folder_stats in file_folders:
            if folder_name in folders:
                # Add folder stats
                folders[folder_name].append(file_folder_stats)
                # Update folder summary
                folders[folder_name][0][2] += file_folder_stats[2]
                folders[folder_name][0][3] += file_folder_stats[3]
                folders[folder_name][0][4] += file_folder_stats[4]
            else:
                folder_summary = [None, 'Summary', file_folder_stats[2],
                                    file_folder_stats[3], file_folder_stats[4]]
                folders[folder_name] = [folder_summary, file_folder_stats]

    ## Calculate ratios
    folders = [('Summary', folders.pop('Summary'))] + sorted(folders.items(), key=lambda x: x[0])
    for folder, file_stats in folders:
        # Insert N/A if histo is missing
        if len(file_stats) != len(files)+1:
            for i, file_ in enumerate(files):
                if file_[0] != file_stats[i][0]:
                    file_stats = file_stats[:i] + [[None, "N/A", None, None, None]] + file_stats[i:]
        # Count the ratios
        for i, stats in enumerate(file_stats):
            stats[2], stats[3], stats[4] = get_percentage(*stats[2:5])
    context['folders'] = folders


    ## Select Summary Barchart, Detailed Barchart
    for folder in folders:
        print folder
    #   detailed_ratios: (name, success_ratio)
    #   summary_ratios: (name, success_ratio)


    ## Summary Barchart
    # TODO: optimise not to fetch from DB again.
    c.execute('''SELECT name, from_histogram_id, till_histogram_id FROM Directory
                 WHERE parent_id IN (SELECT directory_id FROM RootFileComparison
                 WHERE release_comparison_id = ?)''', (release_comp_id,))
    lvl3_dir_ranges = c.fetchall()

    cum_lvl3_dir_ranges = dict()
    for name, from_id, till_id in lvl3_dir_ranges:
        if name in cum_lvl3_dir_ranges:
            cum_lvl3_dir_ranges[name].append((from_id, till_id))
        else:
            cum_lvl3_dir_ranges[name] = [(from_id, till_id)]

    # Fetch stats
    summary_stats = dict()
    detailed_stats = dict()
    for name, ranges in cum_lvl3_dir_ranges.iteritems():
        successes, nulls, fails = get_stats(c, threshold, ranges)
        if name in detailed_stats:
            detailed_stats[name][0] += successes
            detailed_stats[name][1] += nulls
            detailed_stats[name][2] += fails
        else:
            detailed_stats[name] = [successes, nulls, fails]
        if name in renaming:
            if renaming[name] in summary_stats:
                summary_stats[renaming[name]][0] += successes
                summary_stats[renaming[name]][1] += nulls
                summary_stats[renaming[name]][2] += fails
            else:
                summary_stats[renaming[name]] = [successes, nulls, fails]

    # Calculate ratio
    summary_ratios = []
    for name, stats in summary_stats.iteritems():
        total = sum(stats)
        if total:
            ratio = float(stats[0]) / sum(stats)
            summary_ratios.append((name, ratio))
    detailed_ratios = []
    for name, stats in detailed_stats.iteritems():
        total = sum(stats)
        if total:
            ratio = float(stats[0]) / sum(stats)
            detailed_ratios.append((name, ratio))

    context['summary_ratios'] = sorted(summary_ratios, key=lambda x: x[0])
    context['detailed_ratios'] = sorted(detailed_ratios, key=lambda x: x[0])
    return context


def get_directory_summary_stats(c, url_args, file_id, threshold):
    '''Returns context for ``directory_summary.html`` template.'''
    context = dict()
    c.execute('''SELECT directory_id, filename1, filename2 FROM RootFileComparison
                 WHERE id = ?''', (file_id,))
    dir_id, f1, f2 = c.fetchone()
    context['release1'] = get_release(f1)
    context['release2'] = get_release(f2)
    if not url_args:
        dir_name = get_dataset_name(f1)
    else:
        #### Select DQMData/Run directory.
        directory_names = []

        for dir_name in url_args:
            c.execute('''SELECT id, name FROM Directory WHERE name = ? AND
                    parent_id = ?''', (dir_name, dir_id))
            dir_id, name = c.fetchone()
            directory_names.append(name)
        context['parent_name'] = '/'.join(directory_names)

    ## Select stats
    c.execute('''SELECT from_histogram_id, till_histogram_id FROM
                 Directory WHERE id = ?''', (dir_id,))
    ranges = c.fetchone()
    successes, nulls, fails = get_stats(c, threshold, (ranges,))
    success, null, fail = get_percentage(successes, nulls, fails)
    context.update({
            'successes': successes, 'nulls': nulls, 'fails': fails,
            'success': success, 'null': null, 'fail': fail,
            'total': successes + nulls + fails, 'dir_name': dir_name
        })
    # subdirs: name, total, success, fail, null
    c.execute('''SELECT name, from_histogram_id, till_histogram_id FROM Directory
                 WHERE parent_id = ?''', (dir_id,))
    subdirs = c.fetchall()
    subdir_stats = []
    for name, from_id, till_id in subdirs:
        successes, nulls, fails = get_stats(c, threshold, [(from_id, till_id,)])
        success, null, fail = get_percentage(successes, nulls, fails)
        subdir_stats.append((name, successes + nulls + fails, successes,
                             nulls, fails, success, null, fail))
    context['subdirs'] = sorted(subdir_stats, key=lambda x: x[4], reverse=True)

    # histograms: name, p_value
    c.execute('''SELECT name, p_value FROM HistogramComparison
                 WHERE directory_id = ?''', (dir_id,))
    failed_histos = []
    successful_histos = []
    null_histos = []
    for name, p_value in c.fetchall():
        path = quote('%s/%s' % ('/'.join(url_args), name))
        url1 = get_img_url(path, f1)
        url2 = get_img_url(path, f2)
        overlay = get_img_url(path, f1, f2)
        if p_value < 0:
            null_histos.append((name, p_value, url1, url2, overlay))
        elif p_value <= threshold:
            failed_histos.append((name, p_value, url1, url2, overlay))
        else:
            successful_histos.append((name, p_value, url1, url2, overlay))

    context['failed_histos'] = sorted(failed_histos, key=lambda x: x[1], reverse=True)
    context['null_histos'] = null_histos
    context['successful_histos'] = sorted(successful_histos, key=lambda x: x[1], reverse=True)
    return context

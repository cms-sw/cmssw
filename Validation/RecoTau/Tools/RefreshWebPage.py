#!/usr/bin/env python

import os
import glob
import sys
import re
import web_templates as templates
from optparse import OptionParser
from Validation.RecoTau.ValidationOptions_cff import allowedOptions

__author__  = "Mauro Verzetti (mauro.verzetti@cern.ch)"
__doc__ = """Script to update the web-page to show newly updated results. First upload the TauID directory in the proper location in $PastResults"""

parser = OptionParser(description=__doc__)
parser.add_option('--test',action="store_true", dest="test", default=False, help="used to test/debug the script. The html file are written locally")
(options,crap) = parser.parse_args()

def unpackRelease(relName):
    m = re.match(r'^CMSSW_(?P<one>[0-9]+)_(?P<two>[0-9]+)_(?P<three>[0-9]+)(?:_pre(?P<four>[0-9]+))?',relName)
    if m:
        prev = int(m.group('four')) if m.group('four') else 1000000
        return (m.group('one'),m.group('two'),m.group('three'),prev)
    return (0,0)

try:
    webDir = os.environ['PastResults']
except:
    print 'Run cmsenv and source UtilityCommands.(c)sh first!'
    sys.exit(0)

webDir           += '/'
webDir_subdirs    = filter( lambda x: os.path.isdir(webDir+x), os.listdir( webDir ) )
official_releases = sorted( filter( lambda x: re.findall(r'^CMSSW_[0-9]+_[0-9]+_[0-9]+(?:_pre[0-9]+)?$',x), webDir_subdirs), key=unpackRelease)
special_releases  = sorted( filter( lambda x: re.findall(r'^CMSSW_[0-9]+_[0-9]+_[0-9]+',x) and not x in official_releases, webDir_subdirs), key=unpackRelease)
custom_made       = [d for d in webDir_subdirs if not d in official_releases and not d in special_releases]


official_releases_links = ''.join([templates.create_main_list_element(d) for d in official_releases])
special_releases_links  = ''.join([templates.create_main_list_element(d) for d in special_releases])
custom_made_links       = ''.join([templates.create_main_list_element(d) for d in custom_made])
main_web_page           = templates.main_page_template % (official_releases_links, special_releases_links, custom_made_links)
main_web_page_path      = webDir+'index.html' if not options.test else 'index.html'
main_web_page_html      = open(main_web_page_path,'w')
main_web_page_html.write(main_web_page)
main_web_page_html.close()

for rel in official_releases+special_releases:
    tauid_dir = webDir+rel+'/TauID/'
    reldir    = webDir+rel+'/'
    datasets  = filter(lambda x: os.path.isdir(tauid_dir+x) and not x == 'Reference', os.listdir(tauid_dir))
    cfg_file  = glob.glob(tauid_dir+'*/Config/showtags.txt')[0] if glob.glob(tauid_dir+'*/Config/showtags.txt') else None
    config    = open(cfg_file).read() if cfg_file else 'NO CONFIGURATION AVAILABLE!'
    data_html = []
    for dataset in datasets:
        dname    = dataset.split('_')[0]
        if not dname in allowedOptions['eventType']:
            continue
        pics     = [path.split(rel+'/')[1] for path in glob.glob(tauid_dir+dataset+'/*.png')]
        roots    = glob.glob(tauid_dir+dataset+'/*.root')[0]
        rootf    = (roots).split(rel+'/')[1]
        ref_file = (glob.glob(tauid_dir+'Reference/*'+dname+'.root')[0]).split(rel+'/')[1] if glob.glob(tauid_dir+'Reference/*'+dname+'.root') else None
        source   = 'TauID/'+dataset+'/Config/DataSource_cff.py' if os.path.isfile(tauid_dir+dataset+'/Config/DataSource_cff.py') else None
        dir_link = 'TauID/'+dataset+'/'
        data_html.append(templates.usual_validation_dataset_template(dname, rootf, ref_file, dir_link, pics, source) )
    release_html_page     = templates.usual_validation_template.substitute(THIS_RELEASE=rel,CONFIG=config,DATASETS=''.join(data_html) )
    release_web_page_path = reldir+'index.html' if not options.test else 'index.html'
    release_page_html     = open(release_web_page_path,'w')
    release_page_html.write(release_html_page)
    release_page_html.close()

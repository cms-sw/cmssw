#!/usr/bin/env python

import os, sys
if len(sys.argv) > 1:
  os.environ['DBS_STRATEGY'] = sys.argv[1]

import DQMOffline.EGamma.electronDataDiscovery as dbs
os.environ['TEST_HARVESTED_FILE'] = 'rfio:/castor/cern.ch/cms'+dbs.search()[0]

os.system('root -b -l -q electronWget.C')



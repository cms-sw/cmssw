#!/usr/bin/env python

import sys
import os

import DQMOffline.EGamma.electronDbsDiscovery as dbs
os.environ['TEST_HARVESTED_FILE'] = 'rfio:/castor/cern.ch/cms'+dbs.search()[0]

os.system('root -b -l -q electronWget.C')



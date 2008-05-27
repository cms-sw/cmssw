# ValidationTools : Configuration
#   
# Developers:
#   Victor E. Bazterra
#   Kenneth James Smith
#
# Descrition:
#   Full configuration for ValidationTools

import os

variables = {}

# initial directory
variables['HomeDirectory'] = "/uscmst1b_scratch/lpc1/3DayLifetime/kjsmith/CMSSW_2_0_0/src/MC/GeneratorValidation/"


# web directory
variables['WebDirectory'] = "/afs/fnal.gov/files/expwww/uscms/html/uscms_at_work/physics/lpc/organization/enabling/Validate/"
#variables['WebDirectory'] = "/afs/fnal.gov/files/home/room2/kjsmith/public_html/"
variables['HTTPLocation'] = "http://home.fnal.gov/~kjsmith/"

# Directory where releases/references are stored
variables['ReleaseDirectory'] = "/uscms_data/d1/kjsmith/releases/"

# Set port number
variables["PortNumber"] = 44040

# set root directory
variables['VTRoot'] = variables['HomeDirectory']+"root/"

## String for requesting rss generator in DBS.
variables['RssGenerator'] = 'http://cmsdbs.cern.ch/DBS2_discovery/rssGenerator?'
## String for requesting logical file name for a given site.
variables['LFNForSite'] = 'https://cmsweb.cern.ch/dbs_discovery/getLFNsForSite?'  
## Site location strings for dbs.
variables['CERNSITE'] = 'srm.cern.ch'
variables['FNALSITE'] = 'cmssrm.fnal.gov'


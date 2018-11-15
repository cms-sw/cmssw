#!/usr/bin/env python
#____________________________________________________________
#
#  
#
# A very simple script to read beam spot DB payloads
#
# Francisco Yumiceva
# yumiceva@fnal.gov
#
# Fermilab, 2009
#
#____________________________________________________________

"""
   getBeamSpotDB.py

   A very simple script to retrieve from DB a beam spot payload for a given IOV.
   usage: %prog -t <tag name> -r <run number = 1>
   -a, --auth     = AUTH: Authorization path: \"/afs/cern.ch/cms/DB/conddb\"(default), \"/nfshome0/popcondev/conddb\"
   -d, --destDB   = DESTDB: Destination string for DB connection: \"frontier://PromptProd/CMS_COND_31X_BEAMSPOT\"(default), \"oracle://cms_orcon_prod/CMS_COND_31X_BEAMSPOT\", \"sqlite_file:mysqlitefile.db\"
   -g, --globaltag= GLOBALTAG: Name of Global tag. If this is provided, no need to provide beam spot tags.
   -l, --lumi     = LUMI: Lumi section.
   -r, --run      = RUN: Run number.
   -t, --tag      = TAG: Name of Beam Spot DB tag.
   
   Francisco Yumiceva (yumiceva@fnal.gov)
   Fermilab 2010
   
"""
from __future__ import print_function


import sys,os, re
import commands
import six

#_______________OPTIONS________________
import optparse

USAGE = re.compile(r'(?s)\s*usage: (.*?)(\n[ \t]*\n|$)')

def nonzero(self): # will become the nonzero method of optparse.Values
    "True if options were given"
    for v in six.itervalues(self.__dict__):
        if v is not None: return True
    return False

optparse.Values.__nonzero__ = nonzero # dynamically fix optparse.Values

class ParsingError(Exception): pass

optionstring=""

def exit(msg=""):
    raise SystemExit(msg or optionstring.replace("%prog",sys.argv[0]))

def parse(docstring, arglist=None):
    global optionstring
    optionstring = docstring
    match = USAGE.search(optionstring)
    if not match: raise ParsingError("Cannot find the option string")
    optlines = match.group(1).splitlines()
    try:
        p = optparse.OptionParser(optlines[0])
        for line in optlines[1:]:
            opt, help=line.split(':')[:2]
            short,long=opt.split(',')[:2]
            if '=' in opt:
                action='store'
                long=long.split('=')[0]
            else:
                action='store_true'
            p.add_option(short.strip(),long.strip(),
                         action = action, help = help.strip())
    except (IndexError,ValueError):
        raise ParsingError("Cannot parse the option string correctly")
    return p.parse_args(arglist)



# lumi tools CondCore/Utilities/python/timeUnitHelper.py
def pack(high,low):
    """pack high,low 32bit unsigned int to one unsigned 64bit long long
       Note:the print value of result number may appear signed, if the sign bit is used.
    """
    h=high<<32
    return (h|low)

def unpack(i):
    """unpack 64bit unsigned long long into 2 32bit unsigned int, return tuple (high,low)
    """
    high=i>>32
    low=i&0xFFFFFFFF
    return(high,low)

def unpackLumiid(i):
    """unpack 64bit lumiid to dictionary {'run','lumisection'}
    """
    j=unpack(i)
    return {'run':j[0],'lumisection':j[1]}


if __name__ == '__main__':


    # COMMAND LINE OPTIONS
    #################################
    option,args = parse(__doc__)
    if not args and not option: exit()

    tagname = ''
    globaltag = ''

    if ((option.tag and option.globaltag)) == False: 
        print(" NEED to provide beam spot DB tag name, or global tag")
        exit()
    elif option.tag:
        tagname = option.tag
    elif option.globaltag:
        globaltag = option.globaltag
        cmd = 'cmscond_tagtree_list -c frontier://cmsfrontier.cern.ch:8000/Frontier/CMS_COND_31X_GLOBALTAG -P /afs/cern.ch/cms/DB/conddb -T '+globaltag+' | grep BeamSpot'
        outcmd = commands.getstatusoutput( cmd )
        atag = outcmd[1].split()
        atag = atag[2]
        tagname = atag.replace("tag:","")
        print(" Global tag: "+globaltag+" includes the beam spot tag: "+tagname)

    iov_since = ''
    iov_till = ''
    destDB = 'frontier://PromptProd/CMS_COND_31X_BEAMSPOT'
    if option.destDB:
        destDB = option.destDB

    auth = '/afs/cern.ch/cms/DB/conddb'
    if option.auth:
        auth = option.auth

    run = '1'
    if option.run:
        run = option.run
    lumi = '1'
    if option.lumi:
        lumi = option.lumi

    #sqlite_file = "sqlite_file:"+ tagname +".db"


    ##### READ 

    #print "read back sqlite file to check content ..."

    readdb_out = "readDB_"+tagname+".py"

    rnewfile = open(readdb_out,'w')

    rnewfile.write('''
import FWCore.ParameterSet.Config as cms

process = cms.Process("readDB")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("CondCore.DBCommon.CondDBSetup_cfi")

process.BeamSpotDBSource = cms.ESSource("PoolDBESSource",
                                        process.CondDBSetup,
                                        toGet = cms.VPSet(cms.PSet(
    record = cms.string('BeamSpotObjectsRcd'),
''')
    rnewfile.write('tag = cms.string(\''+tagname+'\')\n')
    rnewfile.write(')),\n')
    rnewfile.write('connect = cms.string(\''+destDB+'\')\n')

    #connect = cms.string('sqlite_file:Early900GeVCollision_7p4cm_STARTUP_mc.db')
    #connect = cms.string('oracle://cms_orcoff_prod/CMS_COND_31X_BEAMSPOT')
    #connect = cms.string('frontier://PromptProd/CMS_COND_31X_BEAMSPOT')
    rnewfile.write('''
                                        )

''')
    rnewfile.write('process.BeamSpotDBSource.DBParameters.authenticationPath = cms.untracked.string(\"'+auth + '\")')

    rnewfile.write('''

process.source = cms.Source("EmptySource",
        numberEventsInRun = cms.untracked.uint32(1),
''')
    rnewfile.write('  firstRun = cms.untracked.uint32('+ run + '),\n')
    rnewfile.write('  firstLuminosityBlock = cms.untracked.uint32('+ lumi + ')\n')
    rnewfile.write('''               
)

process.maxEvents = cms.untracked.PSet(
            input = cms.untracked.int32(1)
)
process.beamspot = cms.EDAnalyzer("BeamSpotFromDB")


process.p = cms.Path(process.beamspot)

''')

    rnewfile.close()
    status_rDB = commands.getstatusoutput('cmsRun '+ readdb_out)

    outtext = status_rDB[1]
    print(outtext)

    #### CLEAN up
    #os.system("rm "+ readdb_out)

    print("DONE.\n")

#_________________________________    


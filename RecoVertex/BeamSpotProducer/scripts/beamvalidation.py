#!/usr/bin/env python
#____________________________________________________________
#
#
# A very simple 
#
# Francisco Yumiceva
# yumiceva@fnal.gov
#
# Fermilab, 2010
#
#____________________________________________________________

"""
   bsvalidation

   A very simple script 

   usage: %prog -t <tag name>
   
   
   Francisco Yumiceva (yumiceva@fnal.gov)
   Fermilab 2010
   
"""


import os, string, re, sys, math
import commands, time

#_______________OPTIONS________________
import optparse

USAGE = re.compile(r'(?s)\s*usage: (.*?)(\n[ \t]*\n|$)')

def nonzero(self): # will become the nonzero method of optparse.Values
    "True if options were given"
    for v in self.__dict__.itervalues():
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

#_______________________________

def cmp_tags(a,b):

    tmpa = a.replace("BeamSpotObjects_2009_v","")
    tmpa = tmpa.replace("_offline","")
    tmpb = b.replace("BeamSpotObjects_2009_v","")
    tmpb = tmpb.replace("_offline","")
    
    na = int(tmpa)
    nb = int(tmpb)
    if na < nb: return -1
    if na == nb: return 0
    if na > nb: return 1


if __name__ == '__main__':


    ## Get the latest tags
    queryTags_cmd = "cmscond_list_iov -c frontier://cmsfrontier.cern.ch:8000/Frontier/CMS_COND_31X_BEAMSPOT -P /afs/cern.ch/cms/DB/conddb -a | grep BeamSpotObjects | grep offline"
    
    outcmd = commands.getstatusoutput( queryTags_cmd )
    
    listtags = outcmd[1].split()
    listtags.sort( cmp = cmp_tags )
    listtags.reverte()

    print listtags

    # Get the latest IOVs
    lasttag = listtags[0]

    queryIOVs_cmd = "cmscond_list_iov -c frontier://cmsfrontier.cern.ch:8000/Frontier/CMS_COND_31X_BEAMSPOT -P /afs/cern.ch/cms/DB/conddb -t "+ lasttag

    oudcmd = commands.getstatusoutput( queryIOVs_cmd )

    tmparr = outcmd[1].split('\n')
    TimeType = tmparr[1].split()[1]
    listIOVs = []

    for i in range(0,3):
	tmpline = tmparr[len(tmparr) -2 -i]
	aIOV = []
	aIOV.append( tmpline.split()[0] )
	aIOV.append( tmpline.split()[1] )
	
	listIOVs.append( aIOV )

    listIOVs.reverse()

    print listIOVs



    

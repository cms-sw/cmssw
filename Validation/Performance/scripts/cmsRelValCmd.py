#!/usr/bin/env python
#Parsing the $CMSSW_RELEASE_BASE/src/Configuration/PyReleaseValidation/data/cmsDriver_highstats_htl.txt file
#to look for necessary cmsdriver extra options

import os, re

#Assuming high statistics RelVal cmsDriver command file to be named:
cmsDriverCmd="cmsDriver_highstats_hlt.txt"
#And to be found at:
cmsDriverCmdPath=os.environ["CMSSW_RELEASE_BASE"]+"/src/Configuration/PyReleaseValidation/data"

def get_cmsDriverOptions():
    '''Function returns a string with the cmsDriver.py options relevant to the performance suite used in file cmsDriver_highstats_hlt.txt  in package Configuration/PyReleaseValidation/data\n'''
    filename=os.path.join(cmsDriverCmdPath,cmsDriverCmd)
    cmsDriverOptions=""
    if os.path.exists(filename):
        file=open(filename,"r")
        TTbar=re.compile("TTbar")
        STARTUP=re.compile("STARTUP")
        GENSIM=re.compile("GEN,SIM")
        option=re.compile("^--")
        for line in file.readlines():
            #Always pick the TTbar with IDEAL geometry line to pick up the "standard" options:
            if TTbar.search(line) and STARTUP.search(line) and GENSIM.search(line):
                tokens=line.split()
                #print line
                #print tokens
                for token in tokens:
                    found = option.search(token)
                    #Here we can filter out the options we don't care about:
                    #--relval
                    #--datatier
                    if found and not (found.string == "--relval" or found.string == "--datatier"): 
                        cmsDriverOptions=cmsDriverOptions+found.string+" "+tokens[tokens.index(found.string)+1]+" "
        return cmsDriverOptions
    else:
        print "Could not find file %s!\n"%filename
        return NULL

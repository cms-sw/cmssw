#!/usr/bin/env python

dbPollCommand = "cmscond_list_iov -c frontier://cmsfrontier.cern.ch:8000/Frontier/CMS_COND_31X_BEAMSPOT -t %(tag)s"
tags = { "v25": "BeamSpotObjects_2009_LumiBased_SigmaZ_v25_offline",
         "v26": "BeamSpotObjects_2009_LumiBased_SigmaZ_v26_offline"}
pathTemplate = "SiStripHotComponents_%(tagKey)s"

def getLatestInDB(tag):
    from subprocess import PIPE, Popen
    p = Popen((dbPollCommand%{"tag":tag}).split(),stdout=PIPE)
    rawResult = p.stdout.read()
    result = int([i.split()[0] for i in rawResult.splitlines() if len(i.split()) == 8][-1])
    return result

def getAvailablePayloads(tagName):
    from grub import grub
    result = {}
    for path in grub("%s*.db"%(pathTermplate%{"tagKey":tagName})):
        result[path] = None
    return result

def main():
    for tagName, tag in tags.iteritems():
        latestRun = getLatestInDB( tag)
        print "For DB",tagName, "last IOV is run", latestRun>>32 , " LS",latestRun&0xFFFFFFFF
#         print getAvailablePayloads(tagName)
         
         

if __name__ == '__main__':
    main()
        

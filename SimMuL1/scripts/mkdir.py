from ROOT import *
import time

#_______________________________________________________________________________
def mkdir(name = "triggerEffVsEtaPlots_pu100"):
    """Makes a new output directory according to pattern: [name]_yymmdd_hhmmss"""
    now = list(time.gmtime())
    sec = "%02d" % now[5]
    minu = "%02d" % now[4]
    hour = "%02d" % ((now[3] + 1 ) %24)
    day = "%02d" % now[2]
    month = "%02d" % now[1]
    year = "%02d" % now[0]
    pdir = gSystem.Getenv("PWD") + "/%s_%s%s%s_%s%s%s/"%(name, year, month, day, hour, minu, sec) 
    if gSystem.AccessPathName(pdir)==0:    
        "Directory already exists!"
    else:
        gSystem.MakeDirectory(pdir)
        print "Created output directory: ", pdir
        return pdir

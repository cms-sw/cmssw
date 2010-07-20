#G.Benelli
#This fragment is used to add the SimpleMemoryCheck
#and Timing services output to the log for the Performance Suite profiling.
#It is meant to be used with the cmsDriver.py option
#--customise in the following fashion:
#E.g.
#cmsDriver.py MinBias.cfi -n 50 --step=GEN,SIM --customise=Validation/Performance/TimeMemoryInfo.py >& MinBias_GEN,SIM.log&
#Note there is no need to specify the "python" directory in the path.

import FWCore.ParameterSet.Config as cms
def customise(process):
    #Adding SimpleMemoryCheck service:
    process.SimpleMemoryCheck=cms.Service("SimpleMemoryCheck",
                                          ignoreTotal=cms.untracked.int32(1),
                                          oncePerEventMode=cms.untracked.bool(False))
    #Adding Timing service:
    process.Timing=cms.Service("Timing",
                               summaryOnly=cms.untracked.bool(True))
    
    #Add these 3 lines to put back the summary for timing information at the end of the logfile
    #(needed for TimeReport report)
    process.options = cms.untracked.PSet(
        wantSummary = cms.untracked.bool(True)
        )

    #Add the configuration for the Igprof running to dump profile snapshots:
    process.IgProfService = cms.Service("IgProfService",
        reportFirstEvent            = cms.untracked.int32(1),  #Dump first event for baseline studies
        reportEventInterval         = cms.untracked.int32(100),#Dump every 100 events (101,201,301,401,501)->Will run 501 events for Step2, HLT and FASTSIM (6 profiles+the end of job one)
        reportToFileAtPostEvent     = cms.untracked.string("| gzip -c > IgProf.%I.gz")
        )

    process.genParticles.abortOnUnknownPDGCode = cms.untracked.bool(False)
    
    return(process)

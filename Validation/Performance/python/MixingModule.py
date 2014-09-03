#G.Benelli
#This fragment is used to add the SimpleMemoryCheck
#and Timing services output to the log of the simulation
#performance candles and add pile-up events at the DIGI level using the MixingModule.
#It is meant to be used with the cmsDriver.py option
#--customise in the following fashion:
#E.g.
#./cmsDriver.py MinBias.cfi -n 50 --step=GEN,SIM,DIGI --pileup LowLumiPileUp --customise=Validation/Performance/MixingModule.py >& MINBIAS_GEN,SIM,DIGI_PILEUP.log&

import FWCore.ParameterSet.Config as cms
def customise(process):
    #Renaming the process
    process.__dict__['_Process__name']=process.__dict__['_Process__name']+'-PILEUP'
    #Adding SimpleMemoryCheck service:
    process.SimpleMemoryCheck=cms.Service("SimpleMemoryCheck",
                                          ignoreTotal=cms.untracked.int32(1),
                                          oncePerEventMode=cms.untracked.bool(True))
    #Adding Timing service:
    process.Timing=cms.Service("Timing")

    #Add these 3 lines to put back the summary for timing information at the end of the logfile
    #(needed for TimeReport report)
    if hasattr(process,'options'):
        process.options.wantSummary = cms.untracked.bool(True)
    else:
        process.options = cms.untracked.PSet(
            wantSummary = cms.untracked.bool(True)
        )
            
    #Overwriting the fileNames to be used by the MixingModule
    #when invoking cmsDriver.py with the --PU option
    process.mix.input.fileNames = cms.untracked.vstring('file:../INPUT_PILEUP_EVENTS.root')


    #Add the configuration for the Igprof running to dump profile snapshots:
    process.IgProfService = cms.Service("IgProfService",
                                        reportFirstEvent            = cms.untracked.int32(1),
                                        reportEventInterval         = cms.untracked.int32(50),
                                        reportToFileAtPostEvent     = cms.untracked.string("| gzip -c > IgProf.%I.gz")
                                        )


    return(process)

import FWCore.ParameterSet.Config as cms
def customiseWithTimeMemoryJobReport(process):

    #Adding SimpleMemoryCheck service:
    process.SimpleMemoryCheck=cms.Service("SimpleMemoryCheck",
                                          ignoreTotal=cms.untracked.int32(1),
                                          oncePerEventMode=cms.untracked.bool(False),
                                          jobReportOutputOnly = cms.untracked.bool(True))
    #Adding Timing service:
    process.Timing=cms.Service("Timing",
                               summaryOnly=cms.untracked.bool(True),
                               excessiveTimeThreshold=cms.untracked.double(600))
    
    #Add these 3 lines to put back the summary for timing information at the end of the logfile
    #(needed for TimeReport report)
    if hasattr(process,"options"):
        process.options.wantSummary = cms.untracked.bool(False)
    else:
        process.options = cms.untracked.PSet(
            wantSummary = cms.untracked.bool(False)
            )

    #Silence the final Timing service report
    process.MessageLogger.cerr.TimeReport = cms.untracked.PSet(limit = cms.untracked.int32(0))

    return(process)


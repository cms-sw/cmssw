import FWCore.ParameterSet.Config as cms
def customise(process):

    #Adding SimpleMemoryCheck service:
    process.SimpleMemoryCheck=cms.Service("SimpleMemoryCheck",
                                          ignoreTotal=cms.untracked.int32(1),
                                          oncePerEventMode=cms.untracked.bool(True))
    #Adding Timing service:
    process.Timing=cms.Service("Timing")
    
    #Tweak Message logger to dump G4cout and G4cerr messages in G4msg.log
    #print process.MessageLogger.__dict__
    #Configuring the G4msg.log output
    process.MessageLogger.files = dict(G4msg =  dict(
        noTimeStamps = True
        #First eliminate unneeded output
        ,threshold = 'INFO'
        ,INFO = dict(limit = 0)
        ,FwkReport = dict(limit = 0)
        ,FwkSummary = dict(limit = 0)
        ,Root_NoDictionary = dict(limit = 0)
        ,FwkJob = dict(limit = 0)
        ,TimeReport = dict(limit = 0)
        ,TimeModule = dict(limit = 0)
        ,TimeEvent = dict(limit = 0)
        ,MemoryCheck = dict(limit = 0)
        #TimeModule, TimeEvent, TimeReport are written to LogAsbolute instead of LogInfo with a category
        #so they cannot be eliminated from any destination (!) unless one uses the summaryOnly option
        #in the Timing Service... at the price of silencing the output needed for the TimingReport profiling
        #
        #Then add the wanted ones:
        ,PhysicsList = dict(limit = -1)
        ,G4cout = dict(limit = -1)
        ,G4cerr = dict(limit = -1)
        )
    )
    #Add these 3 lines to put back the summary for timing information at the end of the logfile
    #(needed for TimeReport report)
    if hasattr(process,'options'):
        process.options.wantSummary = cms.untracked.bool(True)
    else:
        process.options.wantSummary = True
        

    return(process)

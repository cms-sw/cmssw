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
    process.MessageLogger.destinations=cms.untracked.vstring('cout'
                                                             ,'cerr'
                                                             ,'G4msg'
                                                             )
    process.MessageLogger.categories=cms.untracked.vstring('FwkJob'
                                                           ,'FwkReport'
                                                           ,'FwkSummary'
                                                           ,'Root_NoDictionary'
                                                           ,'TimeReport'
                                                           ,'TimeModule'
                                                           ,'TimeEvent'
                                                           ,'MemoryCheck'
                                                           ,'PhysicsList'
                                                           ,'G4cout'
                                                           ,'G4cerr'
                                                           )
    #Configuring the G4msg.log output
    process.MessageLogger.G4msg =  cms.untracked.PSet(
        noTimeStamps = cms.untracked.bool(True)
        #First eliminate unneeded output
        ,threshold = cms.untracked.string('INFO')
        ,INFO = cms.untracked.PSet(limit = cms.untracked.int32(0))
        ,FwkReport = cms.untracked.PSet(limit = cms.untracked.int32(0))
        ,FwkSummary = cms.untracked.PSet(limit = cms.untracked.int32(0))
        ,Root_NoDictionary = cms.untracked.PSet(limit = cms.untracked.int32(0))
        ,FwkJob = cms.untracked.PSet(limit = cms.untracked.int32(0))
        ,TimeReport = cms.untracked.PSet(limit = cms.untracked.int32(0))
        ,TimeModule = cms.untracked.PSet(limit = cms.untracked.int32(0))
        ,TimeEvent = cms.untracked.PSet(limit = cms.untracked.int32(0))
        ,MemoryCheck = cms.untracked.PSet(limit = cms.untracked.int32(0))
        #TimeModule, TimeEvent, TimeReport are written to LogAsbolute instead of LogInfo with a category
        #so they cannot be eliminated from any destination (!) unless one uses the summaryOnly option
        #in the Timing Service... at the price of silencing the output needed for the TimingReport profiling
        #
        #Then add the wanted ones:
        ,PhysicsList = cms.untracked.PSet(limit = cms.untracked.int32(-1))
        ,G4cout = cms.untracked.PSet(limit = cms.untracked.int32(-1))
        ,G4cerr = cms.untracked.PSet(limit = cms.untracked.int32(-1))
        )

    #Add these 3 lines to put back the summary for timing information at the end of the logfile
    #(needed for TimeReport report)
    process.options = cms.untracked.PSet(
        wantSummary = cms.untracked.bool(True)
        )

    return(process)

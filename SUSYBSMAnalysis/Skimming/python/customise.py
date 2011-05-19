import FWCore.ParameterSet.Config as cms

def customise(process):
    process.MessageLogger.cerr.FwkReport.reportEvery = 100
    process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

    import FWCore.ParameterSet.SequenceTypes
    for p in process.schedule:
        if (p.__class__==FWCore.ParameterSet.SequenceTypes.EndPath):
            process.schedule.remove( p )
            continue
    for p in process.schedule:
        if (p.__class__==FWCore.ParameterSet.SequenceTypes.EndPath):
            process.schedule.remove( p )
    for p in process.schedule:
        if (p.__class__==FWCore.ParameterSet.SequenceTypes.EndPath):
            process.schedule.remove( p )
    process.outAll = cms.OutputModule(
        "PoolOutputModule",
        fileName = cms.untracked.string('skimAll.root'),
        SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring()
        ),
        outputCommands = cms.untracked.vstring('drop *','keep *_TriggerResults_*_*')                         
        )
    process.outAllEndPath = cms.EndPath(process.outAll)
    process.schedule.append(process.outAllEndPath)
    return(process)

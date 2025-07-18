import FWCore.ParameterSet.Config as cms

def customise_min_source_output(process, nEventsToCache=10):
    from IOPool.Input.modules import RepeatingCachedRootSource
    from FWCore.Modules.modules import AsciiOutputModule

    process.source = RepeatingCachedRootSource(fileName = process.source.fileNames[0],
                                               repeatNEvents = nEventsToCache)

    for k,v in process.outputModules_().items():
        if v.type_() == 'PoolOutputModule':
            setattr(process,k, AsciiOutputModule(verbosity= 0, outputCommands = v.outputCommands))
            if hasattr(k,'SelectEvents'):
                getattr(process,k).SelectEvents = k.SelectEvents

    #decrease messages as events are processed quickly
    process.MessageLogger.cerr.FwkReport.reportEvery = 100

    #avoid warning messages each event if running on empty events
    if hasattr(process, 'manystripclus53X'):
        process.manystripclus53X.multiplicityConfig = dict(firstMultiplicityConfig=dict(warnIfModuleMissing=cms.untracked.bool(False)),
                                                           secondMultiplicityConfig=dict(warnIfModuleMissing=cms.untracked.bool(False)))
    if hasattr(process, 'toomanystripclus53X'):
        process.toomanystripclus53X.multiplicityConfig = dict(firstMultiplicityConfig=dict(warnIfModuleMissing=cms.untracked.bool(False)),
                                                              secondMultiplicityConfig=dict(warnIfModuleMissing=cms.untracked.bool(False)))

    return process

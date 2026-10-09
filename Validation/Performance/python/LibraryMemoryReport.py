import FWCore.ParameterSet.Config as cms
def customise(process):
    process.LibraryMemoryReport = cms.Service("LibraryMemoryReport")
    return(process)

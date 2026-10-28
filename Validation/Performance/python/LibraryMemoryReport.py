import FWCore.ParameterSet.Config as cms
def customiseWithLibraryMemoryReport(process):
    process.LibraryMemoryReport = cms.Service("LibraryMemoryReport")
    return(process)

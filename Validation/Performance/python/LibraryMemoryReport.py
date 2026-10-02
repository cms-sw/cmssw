import FWCore.ParameterSet.Config as cms
def customise(process):
    process.LibraryMemoryReport = cms.Service("libraryMemoryReport",
                                   fileName=cms.untracked.string("libraryMemoryReport.log")
                                   )
    return(process)

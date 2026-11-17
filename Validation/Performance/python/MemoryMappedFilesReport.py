import FWCore.ParameterSet.Config as cms
def customiseWithMemoryMappedFilesReport(process):
    from FWCore.Services.modules import MemoryMappedFilesReport
    process.add_(MemoryMappedFilesReport(allModules = True, collectPaths = True))
    return process


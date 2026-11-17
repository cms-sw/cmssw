import FWCore.ParameterSet.Config as cms
from Validation.Performance.python import MemoryMappedFilesReport
def customiseWithMemoryMappedFilesReport(process):
    from FWCore.Services.modules import MemoryMappedFilesReport
    process.add_(MemoryMappedFilesReport(allModules = True, collectPaths = True))
    return process


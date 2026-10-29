import FWCore.ParameterSet.Config as cms
def customiseWithLibraryMemoryReport(process):
    from FWCore.Services.modules import LibraryMemoryReport
    process.add_(LibraryMemoryReport(allModules = False))
    return process


#! /usr/bin/env cmsRun
import glob
import FWCore.ParameterSet.Config as cms

inputFiles = cms.untracked.vstring()
inputFiles.extend( [('file:%s ' % name) for name in glob.glob('files/*.root')] )
inputFiles.sort()
print inputFiles

process = cms.Process("merge")
process.source = cms.Source("PoolSource",
    fileNames = inputFiles
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('file:material.root')
)

process.end = cms.EndPath(process.out)

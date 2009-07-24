# Runs PFBenchmarkAnalyzer and PFElectronBenchmark on PFElectron sample to
# monitor performance of PFElectron
import sys
import os
sys.path.append('.')
import dbs_discovery
import elec_selection

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("DQMServices.Core.DQM_cfg")


# Source : general definition
process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(),                            
                            secondaryFileNames = cms.untracked.vstring(),
                            noEventSort = cms.untracked.bool(True),
                            duplicateCheckMode = cms.untracked.string('noDuplicateCheck')
                            )

# Update input files
print dbs_discovery.search()
process.source.fileNames.extend(dbs_discovery.search())


process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(-1)
        )


process.load("Validation.RecoParticleFlow.electronBenchmarkGeneric_cff")

# Update the electron selection 
process.gensource.select = elec_selection.selection()
# Update the output file

process.pfElectronBenchmarkGeneric.OutputFile = cms.untracked.string(os.environ['TEST_OUTPUT_FILE'])

process.p =cms.Path(
        process.electronBenchmarkGeneric
        )


#process.out = cms.OutputModule("PoolOutputModule",
#                               outputCommands = cms.untracked.vstring('keep *'),
#                               outputFile = cms.string(os.environ['TEST_OUTPUT_FILE'])
#                               )
#process.outpath = cms.EndPath(process.out)

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.MessageLogger.cerr.FwkReport.reportEvery = 100




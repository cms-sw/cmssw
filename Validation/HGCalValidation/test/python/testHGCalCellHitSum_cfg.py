###############################################################################
# Way to use this:
#   cmsRun testHGCalCellHitSum_cfg.py geometry=V16
#
#   Options for geometry V16, V17
#
###############################################################################
import FWCore.ParameterSet.Config as cms
import os, sys, imp, re
import FWCore.ParameterSet.VarParsing as VarParsing

####################################################################
### SETUP OPTIONS
options = VarParsing.VarParsing('standard')
options.register('geometry',
                 "V17",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "geometry of operations: V16, V17")

### get and parse the command line arguments
options.parseArguments()

import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Era_Phase2C11I13M9_cff import Phase2C11I13M9
process = cms.Process('PROD',Phase2C11I13M9)

print(options)

####################################################################
# Use the options

if (options.geometry == "V16"):
    process.load('Configuration.Geometry.GeometryExtended2026D86Reco_cff')
    geomFile = 'Validation/HGCalValidation/data/wafer_v16.csv'
    hitFile = "file:/eos/user/i/idas/SimOut/DeltaPt/Extended2026D88/step1.root"
else:
    process.load('Configuration.Geometry.GeometryExtended2026D92Reco_cff')
    geomFile = 'Validation/HGCalValidation/data/wafer_v17.csv'
    hitFile = "file:/eos/user/i/idas/SimOut/DeltaPt/Extended2026D92/step1.root"

print("Geometry file: ", geomFile)
print("Hit file: ", hitFile)

process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')

process.load("IOMC.RandomEngine.IOMC_cff")
process.RandomNumberGeneratorService.generator.initialSeed = 456789

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(hitFile)
)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1))

process.load('Validation.HGCalValidation.hgcalCellHitSumEE_cfi')
process.hgcalCellHitSumEE.geometryFileName = geomFile

process.hgcalCellHitSumHEF = process.hgcalCellHitSumEE.clone(
    simhits = 'g4SimHits : HGCHitsHEfront',
    detector = 'HGCalHESiliconSensitive',
    geometryFileName = geomFile
)

process.hgcalCellHitSumHEB = process.hgcalCellHitSumEE.clone(
    simhits = 'g4SimHits : HGCHitsHEback',
    detector = 'HGCalHEScintillatorSensitive',
    geometryFileName = geomFile
)

#process.Tracer = cms.Service("Tracer")

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string('geantoutput.root') )

process.p = cms.Path(process.hgcalCellHitSumEE*process.hgcalCellHitSumHEF*process.hgcalCellHitSumHEB)

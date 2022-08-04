###############################################################################
# Way to use this:
#   cmsRun testHGCalCellHitSum_cfg.py geometry=D92
#
#   Options for geometry D88, D92, D93
#
###############################################################################
import FWCore.ParameterSet.Config as cms
import os, sys, imp, re
import FWCore.ParameterSet.VarParsing as VarParsing

####################################################################
### SETUP OPTIONS
options = VarParsing.VarParsing('standard')
options.register('geometry',
                 "D92",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "geometry of operations: D88, D92, D93")

### get and parse the command line arguments
options.parseArguments()

import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Era_Phase2C11I13M9_cff import Phase2C11I13M9
process = cms.Process('HGCalCellHit',Phase2C11I13M9)

print(options)

####################################################################
# Use the options

if (options.geometry == "D88"):
    process.load('Configuration.Geometry.GeometryExtended2026D88Reco_cff')
    geomFile = 'Validation/HGCalValidation/data/wafer_v16.csv'
    outputFile = 'file:geantoutputD88.root'
elif (options.geometry == "D93"):
    process.load('Configuration.Geometry.GeometryExtended2026D93Reco_cff')
    geomFile = 'Validation/HGCalValidation/data/wafer_v17.csv'
    outputFile = 'file:geantoutputD93.root'
else:
    process.load('Configuration.Geometry.GeometryExtended2026D92Reco_cff')
    geomFile = 'Validation/HGCalValidation/data/wafer_v17.csv'
    outputFile = 'file:geantoutputD92.root'

print("Geometry file: ", geomFile)
print("Output file: ", outputFile)

process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.cerr.FwkReport.reportEvery = 100
if hasattr(process,'MessageLogger'):
    process.MessageLogger.ValidHGCal=dict()

process.load("IOMC.RandomEngine.IOMC_cff")
process.RandomNumberGeneratorService.generator.initialSeed = 456789

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:step1.root')
)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1))

process.load('Validation.HGCalValidation.hgcalCellHitSum_cfi')

process.hgcalCellHitSumEE = process.hgcalCellHitSum.clone(
    geometryFileName = geomFile
)

process.hgcalCellHitSumHEF = process.hgcalCellHitSum.clone(
    simhits = ('g4SimHits', 'HGCHitsHEfront'),
    detector = 'HGCalHESiliconSensitive',
    geometryFileName = geomFile
)

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string(outputFile) )

process.p = cms.Path(process.hgcalCellHitSumEE*process.hgcalCellHitSumHEF)

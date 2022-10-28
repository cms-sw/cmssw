###############################################################################
# Way to use this:
#   cmsRun testHGCalCellHitSum_cfg.py geometry=D92 layers=1 type=mu
#
#   Options for geometry D88, D92, D93
#               layers '1', '1,2', any combination from 1..47           
#               type mu, tt
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

options.register('layers',
                 "1",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "For single layer use 'layers=3' (default is 'layers=1'); or for multiple layers use 'layers=1,27,41,46'; or for all layers use 'layers=1-47'. Note that the size may increase by ~10 times in memory usage and ~50 times in file volume if 'all layers' option is applied.")

options.register('type',
                 "mu",
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "type of operations: mu, tt")

### get and parse the command line arguments
options.parseArguments()

import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9
process = cms.Process('HGCalCellHit',Phase2C17I13M9)

print(options)

####################################################################
# Use the options

loadFile = "Configuration.Geometry.GeometryExtended2026" + options.geometry + "Reco_cff"
inputFile = "file:step1" + options.geometry + options.type + ".root"
outputFile = "file:geantoutput" + options.geometry + options.type + ".root"

if (options.geometry == "D88"):
    geomFile = 'Validation/HGCalValidation/data/wafer_v16.csv'
elif (options.geometry == "D93"):
    geomFile = 'Validation/HGCalValidation/data/wafer_v17.csv'
else:
    geomFile = 'Validation/HGCalValidation/data/wafer_v17.csv'

print("Geometry file: ", loadFile)
print("Wafer file:    ", geomFile)
print("Input file:    ", inputFile)
print("Output file:   ", outputFile)

process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load(loadFile)
process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.cerr.FwkReport.reportEvery = 100
if hasattr(process,'MessageLogger'):
    process.MessageLogger.ValidHGCal=dict()

process.load("IOMC.RandomEngine.IOMC_cff")
process.RandomNumberGeneratorService.generator.initialSeed = 456789

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(inputFile)
)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1))

process.load('Validation.HGCalValidation.hgcalCellHitSum_cff')

process.hgcalCellHitSumEE.geometryFileName = geomFile
process.hgcalCellHitSumHEF.geometryFileName = geomFile
process.hgcalCellHitSumHEB.geometryFileName = geomFile

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string(outputFile) )

process.p = cms.Path(process.hgcalCellHitSumEE*process.hgcalCellHitSumHEF*process.hgcalCellHitSumHEB)

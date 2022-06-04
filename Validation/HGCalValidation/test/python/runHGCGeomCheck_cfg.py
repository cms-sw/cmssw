###############################################################################
# Way to use this:
#   cmsRun runHGCGeomCheck_cfg.py geometry=D77
#
#   Options for geometry D49, D68, D77, D83, D84, D88, D92
#
###############################################################################
import FWCore.ParameterSet.Config as cms
import os, sys, imp, re
import FWCore.ParameterSet.VarParsing as VarParsing

####################################################################
### SETUP OPTIONS
options = VarParsing.VarParsing('standard')
options.register('geometry',
                 "D86",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "geometry of operations: D49, D68, D84, D77, D83, D88, D92")

### get and parse the command line arguments
options.parseArguments()

print(options)

####################################################################
# Use the options

if (options.geometry == "D49"):
    from Configuration.Eras.Era_Phase2C9_cff import Phase2C9
    process = cms.Process('PROD',Phase2C9)
    process.load('Configuration.Geometry.GeometryExtended2026D49_cff')
    process.load('Configuration.Geometry.GeometryExtended2026D49Reco_cff')
    inFile = 'file:testHGCalSimWatcherV11.root'
    outFile = 'hgcGeomCheckD49.root'
elif (options.geometry == "D68"):
    from Configuration.Eras.Era_Phase2C12_cff import Phase2C12
    process = cms.Process('PROD',Phase2C12)
    process.load('Configuration.Geometry.GeometryExtended2026D68_cff')
    process.load('Configuration.Geometry.GeometryExtended2026D68Reco_cff')
    inFile = 'file:testHGCalSimWatcherV12.root'
    outFile = 'hgcGeomCheckD68.root'
elif (options.geometry == "D83"):
    from Configuration.Eras.Era_Phase2C11M9_cff import Phase2C11M9
    process = cms.Process('PROD',Phase2C11M9)
    process.load('Configuration.Geometry.GeometryExtended2026D83_cff')
    process.load('Configuration.Geometry.GeometryExtended2026D83Reco_cff')
    inFile = 'file:testHGCalSimWatcherV15.root'
    outFile = 'hgcGeomCheckD83.root'
elif (options.geometry == "D84"):
    from Configuration.Eras.Era_Phase2C11_cff import Phase2C11
    process = cms.Process('PROD',Phase2C11)
    process.load('Configuration.Geometry.GeometryExtended2026D84_cff')
    process.load('Configuration.Geometry.GeometryExtended2026D84Reco_cff')
    inFile = 'file:testHGCalSimWatcherV13.root'
    outFile = 'hgcGeomCheckD84.root'
elif (options.geometry == "D88"):
    from Configuration.Eras.Era_Phase2C11M9_cff import Phase2C11M9
    process = cms.Process('PROD',Phase2C11M9)
    process.load('Configuration.Geometry.GeometryExtended2026D88_cff')
    process.load('Configuration.Geometry.GeometryExtended2026D88Reco_cff')
    inFile = 'file:testHGCalSimWatcherV16.root'
    outFile = 'hgcGeomCheckD88.root'
elif (options.geometry == "D92"):
    from Configuration.Eras.Era_Phase2C11M9_cff import Phase2C11M9
    process = cms.Process('PROD',Phase2C11M9)
    process.load('Configuration.Geometry.GeometryExtended2026D92_cff')
    process.load('Configuration.Geometry.GeometryExtended2026D92Reco_cff')
    inFile = 'file:testHGCalSimWatcherV17.root'
    outFile = 'hgcGeomCheckD92.root'
else:
    from Configuration.Eras.Era_Phase2C11M9_cff import Phase2C11M9
    process = cms.Process('PROD',Phase2C11M9)
    process.load('Configuration.Geometry.GeometryExtended2026D77_cff')
    process.load('Configuration.Geometry.GeometryExtended2026D77Reco_cff')
    inFile = 'file:testHGCalSimWatcherV14.root'
    outFile = 'hgcGeomCheckD77.root'

print("Input file: ", inFile)
print("Output file: ", outFile)

process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')    
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic_T21', '')

process.MessageLogger.cerr.FwkReport.reportEvery = 5
if hasattr(process,'MessageLogger'):
    process.MessageLogger.HGCalValid=dict()
    process.MessageLogger.HGCalGeom=dict()

process.MessageLogger.cerr.FwkReport.reportEvery = 100
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(inFile)
)

process.load('Validation.HGCalValidation.hgcGeomCheck_cff')

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string(outFile),
                                   closeFileFast = cms.untracked.bool(True)
)

SimpleMemoryCheck = cms.Service("SimpleMemoryCheck",ignoreTotal = cms.untracked.int32(1) )

process.p = cms.Path(process.hgcGeomCheck)



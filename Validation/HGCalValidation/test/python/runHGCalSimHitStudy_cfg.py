###############################################################################
# Way to use this:
#   cmsRun runHGCalSimHitStudy_cfg.py geometry=D110
#
#   Options for geometry D98, D99, D103, D104, D105, D106, D107, D108, D109
#                        D110, D111, D112, D113, D114, D115
#
###############################################################################
import FWCore.ParameterSet.Config as cms
import os, sys, importlib, re
import FWCore.ParameterSet.VarParsing as VarParsing

####################################################################
### SETUP OPTIONS
options = VarParsing.VarParsing('standard')
options.register('geometry',
                 "D110",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "geometry of operations: D98, D99, D103, D104, D105, D106, D107, D108, D109, D110, D111, D112, D113, D114, D115")

### get and parse the command line arguments
options.parseArguments()

print(options)

####################################################################
# Use the options
if (options.geometry == "D115"):
    from Configuration.Eras.Era_Phase2C20I13M9_cff import Phase2C20I13M9
    process = cms.Process('SimHit',Phase2C20I13M9)
elif (options.geometry == "D104"):
    from Configuration.Eras.Era_Phase2C22I13M9_cff import Phase2C22I13M9
    process = cms.Process('SimHit',PhaseC22I13M9)
elif (options.geometry == "D106"):
    from Configuration.Eras.Era_Phase2C22I13M9_cff import Phase2C22I13M9
    process = cms.Process('SimHit',PhaseC22I13M9)
elif (options.geometry == "D109"):
    from Configuration.Eras.Era_Phase2C22I13M9_cff import Phase2C22I13M9
    process = cms.Process('SimHit',PhaseC22I13M9)
elif (options.geometry == "D111"):
    from Configuration.Eras.Era_Phase2C22I13M9_cff import Phase2C22I13M9
    process = cms.Process('SimHit',PhaseC22I13M9)
elif (options.geometry == "D112"):
    from Configuration.Eras.Era_Phase2C22I13M9_cff import Phase2C22I13M9
    process = cms.Process('SimHit',PhaseC22I13M9)
elif (options.geometry == "D113"):
    from Configuration.Eras.Era_Phase2C22I13M9_cff import Phase2C22I13M9
    process = cms.Process('SimHit',PhaseC22I13M9)
else:
    from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9
    process = cms.Process('SimHit',Phase2C17I13M9)

geomFile = "Configuration.Geometry.GeometryExtended2026" + options.geometry + "Reco_cff"
fileInput = "file:step1" + options.geometry + "tt.root"
fileName = "hgcSimHit" + options.geometry + "tt.root"

print("Geometry file: ", geomFile)
print("Input file:    ", fileInput)
print("Output file:   ", fileName)

process.load(geomFile)
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.EventContent.EventContent_cff")
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Validation.HGCalValidation.hgcSimHitStudy_cfi')
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic_T21', '')

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(fileInput) )

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string(fileName),
                                   closeFileFast = cms.untracked.bool(True)
                                   )

process.p = cms.Path(process.hgcalSimHitStudy)

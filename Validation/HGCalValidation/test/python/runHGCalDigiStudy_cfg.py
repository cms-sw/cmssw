###############################################################################
# Way to use this:
#   cmsRun runHGCalDigiStudy_cfg.py geometry=D86
#
#   Options for geometry D77, D83, D86
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
                  "geometry of operations: D77, D83, D86")

### get and parse the command line arguments
options.parseArguments()

print(options)

####################################################################
# Use the options

if (options.geometry == "D83"):
    from Configuration.Eras.Era_Phase2C11M9_cff import Phase2C11M9
    process = cms.Process('PROD',Phase2C11M9)
    process.load('Configuration.Geometry.GeometryExtended2026D83_cff')
    process.load('Configuration.Geometry.GeometryExtended2026D83Reco_cff')
    fileName = 'hgcDigiD83tt.root'
elif (options.geometry == "D86"):
    from Configuration.Eras.Era_Phase2C11M9_cff import Phase2C11M9
    process = cms.Process('PROD',Phase2C11M9)
    process.load('Configuration.Geometry.GeometryExtended2026D86_cff')
    process.load('Configuration.Geometry.GeometryExtended2026D86Reco_cff')
    fileName = 'hgcDigiD86tt.root'
else:
    from Configuration.Eras.Era_Phase2C11_cff import Phase2C11
    process = cms.Process('PROD',Phase2C11)
    process.load('Configuration.Geometry.GeometryExtended2026D77_cff')
    process.load('Configuration.Geometry.GeometryExtended2026D77Reco_cff')
    fileName = 'hgcDigiD77tt.root'

print("Output file: ", fileName)

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Validation.HGCalValidation.hgcDigiStudy_cfi')
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic_T21', '')

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
        'file:step2.root',
        )
                            )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string(fileName),
                                   closeFileFast = cms.untracked.bool(True)
                                   )

process.raw2digi_step = cms.Path(process.RawToDigi)
process.analysis_step = cms.Path(process.hgcalDigiStudyEE+
                                 process.hgcalDigiStudyHEF+
                                 process.hgcalDigiStudyHEB)
process.hgcalDigiStudyEE.verbosity = 1
process.hgcalDigiStudyHEF.verbosity = 1
process.hgcalDigiStudyHEB.verbosity = 1

# Schedule definition
process.schedule = cms.Schedule(process.raw2digi_step,process.analysis_step)

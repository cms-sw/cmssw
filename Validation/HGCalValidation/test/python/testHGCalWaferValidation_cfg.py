###############################################################################
# Way to use this:
#   cmsRun testHGCalWaferValidation_cfg.py geometry=D83
#
#   Options for geometry D77, D82, D83, D88, D92
#
###############################################################################
import FWCore.ParameterSet.Config as cms
import os, sys, imp, re
import FWCore.ParameterSet.VarParsing as VarParsing

####################################################################
### SETUP OPTIONS
options = VarParsing.VarParsing('standard')
options.register('geometry',
                 "D83",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "geometry of operations: D77, D82, D83, D88, D92")

### get and parse the command line arguments
options.parseArguments()
print(options)

####################################################################
# Use the options

if (options.geometry == "D77"):
    from Configuration.Eras.Era_Phase2C11M9_cff import Phase2C11M9
    process = cms.Process('TEST',Phase2C11M9)
    process.load('Configuration.Geometry.GeometryExtended2026D77_cff')
    fileName = 'Validation/HGCalValidation/data/geomnew_corrected_360.txt'
elif (options.geometry == "D82"):
    from Configuration.Eras.Era_Phase2C11_cff import Phase2C11
    process = cms.Process('TEST',Phase2C11)
    process.load('Configuration.Geometry.GeometryExtended2026D82_cff')
    fileName = 'Validation/HGCalValidation/data/geomnew_corrected_360.txt'
elif (options.geometry == "D88"):
    from Configuration.Eras.Era_Phase2C11_cff import Phase2C11
    process = cms.Process('TEST',Phase2C11)
    process.load('Configuration.Geometry.GeometryExtended2026D88_cff')
    fileName = 'Validation/HGCalValidation/data/geomnew_corrected_360_V1.txt'
elif (options.geometry == "D92"):
    from Configuration.Eras.Era_Phase2C11_cff import Phase2C11
    process = cms.Process('TEST',Phase2C11)
    process.load('Configuration.Geometry.GeometryExtended2026D92_cff')
    fileName = 'Validation/HGCalValidation/data/geomnew_corrected_360_V2.txt'
else:
    from Configuration.Eras.Era_Phase2C11M9_cff import Phase2C11M9
    process = cms.Process('TEST',Phase2C11M9)
    process.load('Configuration.Geometry.GeometryExtended2026D83_cff')
    fileName = 'Validation/HGCalValidation/data/geomnew_corrected_360.txt'

process.load('Validation.HGCalValidation.hgcalWaferValidation_cfi')
process.hgcalWaferValidation.GeometryFileName = cms.FileInPath(fileName)

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger = cms.Service("MessageLogger",
   destinations   = cms.untracked.vstring('cout'),
   cout           = cms.untracked.PSet(
                        threshold  = cms.untracked.string('INFO')
                        #threshold  = cms.untracked.string('WARNING')
                    ),
)

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))


process.p = cms.Path(process.hgcalWaferValidation)

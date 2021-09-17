###############################################################################
# Way to use this:
#   cmsRun testHGCalWaferValidation_cfg.py geometry=D83
#
#   Options for geometry D49, D68, D77, D83, D84
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
                  "geometry of operations: D49, D68, D84, D77, D83")

### get and parse the command line arguments
options.parseArguments()
#print(options)

####################################################################
# Use the options

if (options.geometry == "D49"):
    from Configuration.Eras.Era_Phase2C9_cff import Phase2C9
    process = cms.Process('TEST',Phase2C9)
    process.load('Configuration.Geometry.GeometryExtended2026D49_cff')
elif (options.geometry == "D68"):
    from Configuration.Eras.Era_Phase2C12_cff import Phase2C12
    process = cms.Process('TEST',Phase2C12)
    process.load('Configuration.Geometry.GeometryExtended2026D68_cff')
elif (options.geometry == "D77"):
    from Configuration.Eras.Era_Phase2C11M9_cff import Phase2C11M9
    process = cms.Process('TEST',Phase2C11M9)
    process.load('Configuration.Geometry.GeometryExtended2026D77_cff')
elif (options.geometry == "D84"):
    from Configuration.Eras.Era_Phase2C11_cff import Phase2C11
    process = cms.Process('TEST',Phase2C11)
    process.load('Configuration.Geometry.GeometryExtended2026D84_cff')
else:
    from Configuration.Eras.Era_Phase2C11M9_cff import Phase2C11M9
    process = cms.Process('TEST',Phase2C11M9)
    process.load('Configuration.Geometry.GeometryExtended2026D83_cff')

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

process.load('Validation.HGCalValidation.hgcalWaferValidation_cfi')
process.hgcalWaferValidation.GeometryFileName = "Validation/HGCalValidation/data/geomnew_corrected_360.txt"

process.p = cms.Path(process.hgcalWaferValidation)

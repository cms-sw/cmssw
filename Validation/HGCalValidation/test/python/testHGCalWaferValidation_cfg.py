###############################################################################
# Way to use this:
#   cmsRun testHGCalWaferValidation_cfg.py geometry=D99
#
#   Options for geometry D98, D99, D108, D94, D103, D104, D106, D109
#
###############################################################################
import FWCore.ParameterSet.Config as cms
import os, sys, importlib, re
import FWCore.ParameterSet.VarParsing as VarParsing

####################################################################
### SETUP OPTIONS
options = VarParsing.VarParsing('standard')
options.register('geometry',
                 "D99",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "geometry of operations: D98, D99, D108, D94, D103, D104, D106, D109")

### get and parse the command line arguments
options.parseArguments()
print(options)

####################################################################
# Use the options
if (options.geometry == "D94"):
    from Configuration.Eras.Era_Phase2C20I13M9_cff import Phase2C20I13M9
    process = cms.Process('Client',Phase2C20I13M9)
elif (options.geometry == "D104"):
    from Configuration.Eras.Era_Phase2C22I13M9_cff import Phase2C22I13M9
    process = cms.Process('Client',PhaseC22I13M9)
elif (options.geometry == "D106"):
    from Configuration.Eras.Era_Phase2C22I13M9_cff import Phase2C22I13M9
    process = cms.Process('Client',PhaseC22I13M9)
elif (options.geometry == "D109"):
    from Configuration.Eras.Era_Phase2C22I13M9_cff import Phase2C22I13M9
    process = cms.Process('Client',PhaseC22I13M9)
else:
    from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9
    process = cms.Process('Client',Phase2C17I13M9)

geomFile = "Configuration.Geometry.GeometryExtended2026" + options.geometry + "Reco_cff"
if (options.geometry == "D98"):
    fileName = 'Validation/HGCalValidation/data/geomnew_corrected_360_V1.txt'
else:
    fileName = 'Validation/HGCalValidation/data/geomnew_corrected_360_V2.txt'

print("Geometry file: ", geomFile)
print("File Name:     ", fileName)

process.load(geomFile)
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
# foo bar baz
# 7HM3RYJAbD9be

###############################################################################
# Way to use this:
#   cmsRun runHGCalRecHitStudy_cfg.py geometry=D82
#
#   Options for geometry D88, D92, D93
#
###############################################################################
import FWCore.ParameterSet.Config as cms
import os, sys, imp, re
import FWCore.ParameterSet.VarParsing as VarParsing

####################################################################
### Run as
## $cmsRun testHGCalMTRecoStudy_cfg.py geometry=D88 layers=1

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
                  "For single layer use 'layers=3' (default is 'layers=1'); or for multiple layers use 'layers=1,27,41,46'; or for all layers use 'layers=1-47'. Note that the size may increase by ~10 times or more in RAM and filesize if 'all layers' option is used.")

### get and parse the command line arguments
options.parseArguments()

import FWCore.ParameterSet.Config as cms

print(options)

####################################################################
# Use the options

fileInput = "file:step3.root"

if (options.geometry == "D88"):
    from Configuration.Eras.Era_Phase2C11I13M9_cff import Phase2C11I13M9
    process = cms.Process('HGCalMTReco',Phase2C11I13M9)
    process.load('Configuration.Geometry.GeometryExtended2026D88Reco_cff')
    outputFile = 'file:recoutputD88.root'
elif (options.geometry == "D93"):
    from Configuration.Eras.Era_Phase2C11I13M9_cff import Phase2C11I13M9
    process = cms.Process('HGCalMTReco',Phase2C11I13M9)
    process.load('Configuration.Geometry.GeometryExtended2026D93Reco_cff')
    outputFile = 'file:recoutputD93.root'
elif (options.geometry == "D92"):
    from Configuration.Eras.Era_Phase2C11I13M9_cff import Phase2C11I13M9
    process = cms.Process('HGCalMTReco',Phase2C11I13M9)
    process.load('Configuration.Geometry.GeometryExtended2026D92Reco_cff')
    outputFile = 'file:recoutputD92.root'
else:
    print("Please select a valid geometry version e.g. D88, D92, D93....")

print("Input file: ", fileInput)
print("Output file: ", outputFile)

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic_T21', '')

process.MessageLogger.cerr.FwkReport.reportEvery = 100

process.source = cms.Source("PoolSource",
    dropDescendantsOfDroppedBranches = cms.untracked.bool(False),
    fileNames = cms.untracked.vstring(fileInput),
    inputCommands = cms.untracked.vstring(
        'keep *',
        'drop l1tTkPrimaryVertexs_L1TkPrimaryVertex__*' # This is to skip this branch causing issue in D88 reco files with older CMSSW <= 12_4_0-pre4
    ),
    secondaryFileNames = cms.untracked.vstring()
)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(options.maxEvents) )

process.load('Validation.HGCalValidation.hgcalMTRecoStudy_cfi')

process.hgcalMTRecoStudyEE = process.hgcalMTRecoStudy.clone(detectorName = 'HGCalEESensitive',
                                                            source = 'HGCalRecHit:HGCEERecHits',
                                                            layerList = options.layers
                                                        )

process.hgcalMTRecoStudyFH = process.hgcalMTRecoStudy.clone(detectorName  = 'HGCalHESiliconSensitive',
                                                            source        = 'HGCalRecHit:HGCHEFRecHits',
                                                            layerList = options.layers
                                                        )

process.hgcalMTRecoStudyBH = process.hgcalMTRecoStudy.clone( detectorName  = 'HGCalHEScintillatorSensitive',
                                                             source        = 'HGCalRecHit:HGCHEBRecHits',
                                                             layerList = options.layers
                                                         )

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string(outputFile),
                                   closeFileFast = cms.untracked.bool(True))

process.p = cms.Path(process.hgcalMTRecoStudyEE+process.hgcalMTRecoStudyFH+process.hgcalMTRecoStudyBH)


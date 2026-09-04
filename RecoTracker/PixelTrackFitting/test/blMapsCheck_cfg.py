# Checks that hltESPBLBFieldMap equals the MagneticField sampled on its lattice and that
# hltESPBLMaterialMap equals the compiled-in D121 table.
#   cmsRun blMapsCheck_cfg.py [mapsFile=blMaps.txt]
import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing
from Configuration.Eras.Era_Phase2C22I13M9_cff import Phase2C22I13M9

options = VarParsing('analysis')
options.register('mapsFile', '', VarParsing.multiplicity.singleton, VarParsing.varType.string,
                 "write both lattices to this text file")
options.parseArguments()

process = cms.Process("BLMAPS", Phase2C22I13M9)
process.load('Configuration.StandardSequences.Accelerators_cff')
process.load('Configuration.Geometry.GeometryExtendedRun4D121Reco_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic_T35', '')
process.load('HLTrigger.Configuration.HLT_75e33.eventsetup.hltESPBLBFieldMap_cfi')
process.load('HLTrigger.Configuration.HLT_75e33.eventsetup.hltESPBLMaterialMap_cfi')

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(1))
process.blMapsCheck = cms.EDAnalyzer("BLMapsCheck", outputFile=cms.untracked.string(options.mapsFile))
process.p = cms.Path(process.blMapsCheck)

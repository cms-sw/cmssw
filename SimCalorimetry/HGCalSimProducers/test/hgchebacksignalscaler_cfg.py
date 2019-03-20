#run with: cmsRun hgchebacksignalscaler_cfg.py doseMap=../data/doseParams_3000fb.txt

import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing
from Configuration.StandardSequences.Eras import eras

options = VarParsing()
options.register ("doseMap", "",  VarParsing.multiplicity.singleton, VarParsing.varType.string)
options.parseArguments()

process = cms.Process("demo",eras.Phase2C4)

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load('Configuration.Geometry.GeometryExtended2023D28Reco_cff')
process.GlobalTag.globaltag = '103X_upgrade2023_realistic_v2'

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )
process.source = cms.Source("EmptySource")

process.plotter = cms.EDAnalyzer("HGCHEbackSignalScalerAnalyzer",
    doseMap  = cms.string( options.doseMap ),
)

process.TFileService = cms.Service("TFileService",
    fileName = cms.string("scalingNumbers.root")
)

process.p = cms.Path(process.plotter)

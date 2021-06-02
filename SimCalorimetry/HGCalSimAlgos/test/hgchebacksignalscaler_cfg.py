#run with: cmsRun hgchebacksignalscaler_cfg.py doseMap=SimCalorimetry/HGCalSimProducers/data/doseParams_3000fb_fluka-3.5.15.9.txt sipmMap=SimCalorimetry/HGCalSimProducers/data/sipmParams_geom-10.txt nPEperMIP=21 doseMapAlgo=0 referenceIdark=0.5

import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing
from Configuration.StandardSequences.Eras import eras

options = VarParsing()
options.register ("doseMap", "",  VarParsing.multiplicity.singleton, VarParsing.varType.string)
options.register ("doseMapAlgo", 0,  VarParsing.multiplicity.singleton, VarParsing.varType.int)
options.register ("sipmMap", "",  VarParsing.multiplicity.singleton, VarParsing.varType.string)
options.register ("nPEperMIP", "",  VarParsing.multiplicity.singleton, VarParsing.varType.int)
options.register ("referenceIdark", 0.5,  VarParsing.multiplicity.singleton, VarParsing.varType.double)
options.parseArguments()

process = cms.Process("demo",eras.Phase2C9)

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load('Configuration.Geometry.{}_cff'.format(options.geometry))
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic', '')

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )
process.source = cms.Source("EmptySource")

process.plotter = cms.EDAnalyzer("HGCHEbackSignalScalerAnalyzer",
                                 doseMap  = cms.string( options.doseMap ),
                                 doseMapAlgo = cms.uint32(options.doseMapAlgo),
                                 sipmMap  = cms.string( options.sipmMap ),
                                 nPEperMIP = cms.uint32( options.nPEperMIP ),
                                 referenceIdark = cms.double( options.referenceIdark ) )

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string("sipmontile_dosemap_output.root")
                               )

process.p = cms.Path(process.plotter)

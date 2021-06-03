#run with: cmsRun hgchebacksignalscaler_cfg.py doseMap=SimCalorimetry/HGCalSimProducers/data/doseParams_3000fb_fluka-3.7.20.txt nPEperMIP=21

import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing
from Configuration.StandardSequences.Eras import eras

options = VarParsing()
options.register("doseMap",   "",  VarParsing.multiplicity.singleton, VarParsing.varType.string)
options.register("nPEperMIP", 21,  VarParsing.multiplicity.singleton, VarParsing.varType.int)
options.register("geom", "GeometryExtended2026D49Reco",  VarParsing.multiplicity.singleton, VarParsing.varType.string)
options.parseArguments()

from Configuration.Eras.Era_Phase2C10_cff import Phase2C10
process = cms.Process('demo',Phase2C10)
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load('Configuration.Geometry.{}_cff'.format(options.geom))

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic', '')

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )
process.source = cms.Source("EmptySource")

process.aged_2020     = cms.EDAnalyzer("HGCHEbackSignalScalerAnalyzer",
                                       doseMap  = cms.string( options.doseMap ),
                                       doseMapAlgo = cms.uint32( 2 ),
                                       sipmMap  = cms.string( 'SimCalorimetry/HGCalSimProducers/data/sipmParams_geom-10.txt' ),
                                       nPEperMIP = cms.uint32( options.nPEperMIP ),
                                       referenceIdark = cms.double( -1 ) )
process.aged_2020_noAreaScaling = process.aged_2020.clone( doseMapAlgo = cms.uint32(2+1+4) )
process.aged_2020_noRadScaling  = process.aged_2020.clone( doseMapAlgo = cms.uint32(2+8+16) )
process.aged_2020_noNoise       = process.aged_2020.clone( doseMapAlgo = cms.uint32(2+32) )
process.aged_2021               = process.aged_2020.clone( doseMapAlgo = cms.uint32(2),
                                                           referenceIdark = cms.double(0.5) )

#add tfile service
process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string("sipmontile_dosemap_output.root"))

process.p = cms.Path( 
    process.aged_2020
    *process.aged_2020_noAreaScaling
    *process.aged_2020_noRadScaling
    *process.aged_2020_noNoise
    *process.aged_2021
)


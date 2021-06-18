#run with: cmsRun test/hgchebacksignalscaler_cfg.py doseMap=SimCalorimetry/HGCalSimProducers/data/doseParams_3000fb_fluka-3.7.20.txt geom=GeometryExtended2026D49Reco

import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing
from Configuration.StandardSequences.Eras import eras

options = VarParsing()
options.register("doseMap",   "",  VarParsing.multiplicity.singleton, VarParsing.varType.string)
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

process.ideal = cms.EDAnalyzer("HGCHEbackSignalScalerAnalyzer",
                              doseMap  = cms.string( options.doseMap ),
                              doseMapAlgo = cms.uint32( 1+2+4+8+16+32 ),
                              sipmMap  = cms.string( 'SimCalorimetry/HGCalSimProducers/data/sipmParams_geom-10.txt' ),
                              referenceIdark = cms.double( -1 ) )
process.noise2020 = process.ideal.clone( doseMapAlgo = cms.uint32( 1+2+4+8+16 ) )
process.noise     = process.ideal.clone( doseMapAlgo = cms.uint32( 1+2+4+8+16 ),
                                         referenceIdark = cms.double(0.25) )
process.noise_rad = process.ideal.clone( doseMapAlgo = cms.uint32( 1+2+4 ),
                                         referenceIdark = cms.double(0.25) )
process.noise_atile = process.ideal.clone( doseMapAlgo = cms.uint32( 1+2+8+16 ),
                                           referenceIdark = cms.double(0.25) )
process.noise_radpatile = process.ideal.clone( doseMapAlgo = cms.uint32( 1+2 ),
                                               referenceIdark = cms.double(0.25) )
process.eol = process.ideal.clone( doseMapAlgo = cms.uint32( 2 ),
                                   referenceIdark = cms.double(0.25) )
process.startup = process.ideal.clone( doseMapAlgo = cms.uint32( 2+8+16 ),
                                       referenceIdark = cms.double(0.05) )
process.eol_2020 = process.ideal.clone( doseMapAlgo = cms.uint32( 2 ),
                                        referenceIdark = cms.double(-1) )


#add tfile service
process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string("sipmontile_dosemap_output.root"))

process.p = cms.Path( 
    process.ideal
    *process.noise2020
    *process.noise
    *process.noise_rad
    *process.noise_atile
    *process.noise_radpatile
    *process.eol
    *process.eol_2020
    *process.startup
)


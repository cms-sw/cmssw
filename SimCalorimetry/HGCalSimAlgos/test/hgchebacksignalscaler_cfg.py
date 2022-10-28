#run with: cmsRun test/hgchebacksignalscaler_cfg.py doseMap=SimCalorimetry/HGCalSimProducers/data/doseParams_3000fb_fluka-3.7.20.txt geom=GeometryExtended2026D49Reco

import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing
from Configuration.StandardSequences.Eras import eras

options = VarParsing()
options.register("doseMap",  "SimCalorimetry/HGCalSimProducers/data/doseParams_3000fb_fluka-3.7.20.txt",  VarParsing.multiplicity.singleton, VarParsing.varType.string)
options.register("sipmMap",  "SimCalorimetry/HGCalSimProducers/data/sipmParams_geom-10.txt",  VarParsing.multiplicity.singleton, VarParsing.varType.string)
options.register("geometry", "GeometryExtended2026D49Reco",  VarParsing.multiplicity.singleton, VarParsing.varType.string)
options.parseArguments()

from Configuration.Eras.Era_Phase2C11I13M9_cff import Phase2C11I13M9
process = cms.Process('demo',Phase2C11I13M9)

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load('Configuration.Geometry.{}_cff'.format(options.geometry))

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic', '')

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )
process.source = cms.Source("EmptySource")


#startup: custom sipm map, no dose or fluence
process.startup = cms.EDAnalyzer("HGCHEbackSignalScalerAnalyzer",
                                 doseMap  = cms.string( options.doseMap ),
                                 doseMapAlgo = cms.uint32( 2+8+16 ), 
                                 sipmMap  = cms.string( options.sipmMap ),
                                 referenceIdark = cms.double( 0.125 ) )

#end-of-life (same radiation/sipm maps)
process.eol = process.startup.clone( doseMapAlgo = cms.uint32( 2 ),
                                     referenceIdark = cms.double( 0.5 ) )

#add tfile service
process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string("sipmontile_dosemap_{}.root".format(options.geometry))
)

process.p = cms.Path( 
    process.startup
    *process.eol
)


import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_cff import Run3
process = cms.Process('Dump',Run3)
#from Configuration.Eras.Era_Run3_dd4hep_cff import Run3_dd4hep
#process = cms.Process('Dump',Run3_dd4hep)

# import of standard configurations
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.Geometry.GeometryExtended2021_cff') 
process.load('Configuration.Geometry.GeometryExtended2021Reco_cff') 
#process.load('Configuration.Geometry.GeometryDD4hepExtended2021_cff') 
#process.load('Configuration.Geometry.GeometryDD4hepExtended2021Reco_cff') 
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.EventContent.EventContent_cff")
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load("SimG4CMS.Calo.EcalSimHitStudy_cfi")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag 
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2021_realistic', '')

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring('file:singleElectron_ddd.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string('ecalHitddd.root'),
                                   closeFileFast = cms.untracked.bool(True)
                                   )

process.p = cms.Path(process.EcalSimHitStudy)

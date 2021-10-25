import FWCore.ParameterSet.Config as cms

process = cms.Process("HcalValid")
process.load("DQMServices.Core.DQM_cfg")

#Magnetic Field 		
process.load("Configuration.StandardSequences.MagneticField_cff")

#Geometry
process.load("Geometry.CMSCommonData.ecalhcalGeometryXML_cfi")
process.load("Geometry.EcalCommonData.ecalSimulationParameters_cff")
process.load("Geometry.HcalCommonData.hcalDDConstants_cff")

process.load("Validation.HcalHits.HcalSimHitStudy_cfi")

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('*'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO'),
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        HcalSim = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        )
    ),
    categories = cms.untracked.vstring('HcalSim'),
    destinations = cms.untracked.vstring('cout')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    debugFlag = cms.untracked.bool(True),
    debugVebosity = cms.untracked.uint32(11),
    fileNames = cms.untracked.vstring('file:simevent.root')
)

process.p1 = cms.Path(process.hcalSimHitStudy)
#process.DQM.collectorHost = ''
process.hcalSimHitStudy.outputFile = 'hcalsimstudy.root'



import FWCore.ParameterSet.Config as cms

process = cms.Process("HcalValid")
process.load("DQMServices.Core.DQM_cfg")

#Magnetic Field 		
process.load("Configuration.StandardSequences.MagneticField_cff")

#Geometry
process.load("Geometry.CMSCommonData.ecalhcalGeometryXML_cfi")

process.load("Validation.HcalHits.HcalHitValidation_cfi")

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('*'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO'),
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        HcalHitValid = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        )
    ),
    categories = cms.untracked.vstring('HcalHitValid'),
    destinations = cms.untracked.vstring('cout')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    debugFlag = cms.untracked.bool(True),
    debugVebosity = cms.untracked.uint32(11),
    fileNames = cms.untracked.vstring('file:simevent_HF.root')
)

process.p1 = cms.Path(process.hcalHitValid)
process.DQM.collectorHost = ''
process.hcalHitValid.outputFile = 'valid_HF.root'


# foo bar baz
# VzGZgLC4lumn0
# eyfDCt351EdtG

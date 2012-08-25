import FWCore.ParameterSet.Config as cms
import os 

process = cms.Process("HcalValid")
process.load("DQMServices.Core.DQM_cfg")

#Magnetic Field 		
process.load("Configuration.StandardSequences.MagneticField_cff")

#Geometry
process.load("Geometry.CMSCommonData.cmsExtendedGeometryXML_cfi")

process.load("Validation.HcalHits.ZdcSimHitStudy_cfi")
process.load("Validation.HcalDigis.ZDCDigiStudy_cfi")

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('*'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO'),
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        ZdcSim = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        )
    ),
    categories = cms.untracked.vstring('ZdcSim'),
    destinations = cms.untracked.vstring('cout')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
#    debugFlag = cms.untracked.bool(True),
#    debugVebosity = cms.untracked.uint32(11),
       fileNames = cms.untracked.vstring('file:simevent.root')
)


print process.ZDCDigiStudy.Verbose
process.p1 = cms.Path(
                     process.ZDCDigiStudy
                     *process.zdcSimHitStudy)
process.DQM.collectorHost = ''
process.zdcSimHitStudy.outputFile = 'zdcStudy.root'
process.ZDCDigiStudy.outputFile='zdcStudy.root'


import FWCore.ParameterSet.Config as cms
process = cms.Process("HcalTestNumberingTest")

process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Geometry.CMSCommonData.cmsExtendedGeometry2017Plan1XML_cfi')
process.load('Geometry.HcalCommonData.hcalDDConstants_cff')

if hasattr(process,'MessageLogger'):
    process.MessageLogger.HcalSim=dict()

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.load('SimG4CMS.Calo.hcalTestNumberingTest_cfi')

process.Timing = cms.Service("Timing")
process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck")

process.p1 = cms.Path(process.hcalTestNumberingTest)

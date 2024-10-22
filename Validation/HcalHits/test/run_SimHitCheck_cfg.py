import FWCore.ParameterSet.Config as cms

process = cms.Process("HcalValid")

#Magnetic Field 		
process.load("Configuration.StandardSequences.MagneticField_cff")

#Geometry
process.load("Configuration.Geometry.GeometryExtended2021Reco_cff")

process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.cerr.FwkReport.reportEvery = 5
if hasattr(process,'MessageLogger'):
    process.MessageLogger.HcalSim=dict()

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:step1_ZMM_ddd.root')
)

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string('hcalDDD2021.root'),
                                   closeFileFast = cms.untracked.bool(True)
                                   )

process.load("Validation.HcalHits.hcalSimHitCheck_cfi")

process.hcalSimHitCheck.outputFile = 'hcalsimcheckDDD.root'
process.hcalSimHitCheck.Verbose = 0

process.p1 = cms.Path(process.hcalSimHitCheck)



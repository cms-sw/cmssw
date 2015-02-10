import FWCore.ParameterSet.Config as cms

process = cms.Process("DigiValidationOnly")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("Configuration.StandardSequences.GeometryRecoDB_cff")

process.load("Configuration.StandardSequences.MagneticField_cff")

process.load("Configuration.StandardSequences.Services_cff")

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

process.load("Validation.TrackerConfiguration.test.RelValSingleMuPt10_cff")

process.load("SimGeneral.MixingModule.mixNoPU_cfi")

#UNCOMMENTGAIN process.load("CalibTracker.Configuration.SiStripGain.SiStripGain_Fake_cff")

#UNCOMMENTGAIN process.load("CalibTracker.Configuration.SiPixelGain.SiPixelGain_Fake_cff")

#UNCOMMENTLA process.load("CalibTracker.Configuration.SiStripLorentzAngle.SiStripLorentzAngle_Fake_cff")

#UNCOMMENTLA process.load("CalibTracker.Configuration.SiPixelLorentzAngle.SiPixelLorentzAngle_Fake_cff")

#UNCOMMENTNOISE process.load("CalibTracker.Configuration.SiStripNoise.SiStripNoise_Fake_APVModePeak_cff")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

process.load("SimTracker.Configuration.SimTracker_cff")

process.load("Validation.TrackerDigis.trackerDigisValidation_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:/tmp/genta/686C53F9-E933-DD11-B74E-001617DC1F70.root')
)

process.Timing = cms.Service("Timing")

process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck")

#UNCOMMENTGAIN process.prefer("SiPixelFakeGainOfflineESSource")
#UNCOMMENTGAIN process.prefer("SiStripGainFakeESSource")
#UNCOMMENTLA process.prefer("SiPixelFakeLorentzAngleESSource")
#UNCOMMENTLA process.prefer("SiStripLAFakeESSource")
#UNCOMMENTNOISE process.prefer("SiStripNoiseFakeESSource")
process.digis = cms.Sequence(process.trDigi*process.trackerDigisValidation)
process.p1 = cms.Path(process.mix*process.digis)
process.GlobalTag.globaltag = 'SCENARIO::All'



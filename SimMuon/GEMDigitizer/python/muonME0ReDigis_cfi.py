import FWCore.ParameterSet.Config as cms

from SimMuon.GEMDigitizer.muonME0DigisPreReco_cfi import me0PreRecoDigiCommonParameters

# Module to create simulated ME0 Pre Reco digis.
simMuonME0ReDigis = cms.EDProducer("ME0ReDigiProducer",
    inputCollection = cms.string('simMuonME0Digis'),
    timeResolution = cms.double(5), # in ns
    minBunch = cms.int32(-5),
    maxBunch = cms.int32(3),
    smearTiming = cms.bool(True),
    discretizeTiming = cms.bool(True),
    radialResolution = cms.double(0.02), # in cm average resolution along radial
    smearRadial = cms.bool(True),
    oldXResolution = cms.double(0.00),
    oldYResolution = cms.double(0.00),
    newXResolution = cms.double(0.03),
    newYResolution = cms.double(2.50),
    discretizeX = cms.bool(False),
    discretizeY = cms.bool(True),
    verbose = cms.bool(False),
    reDigitizeOnlyMuons = cms.bool(False),
    reDigitizeNeutronBkg = cms.bool(True),
    rateFact = me0PreRecoDigiCommonParameters.rateFact, # This must be synchronized with the default digitizer
    instLumiDefault = me0PreRecoDigiCommonParameters.instLumi, # This must be synchronized with the default digitizer
    instLumi = cms.double(7.5), # in units of 1E34 cm^-2 s^-1
)

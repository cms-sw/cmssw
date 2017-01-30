import FWCore.ParameterSet.Config as cms

me0PreRecoDigiCommonParameters = cms.PSet(
    inputCollection = cms.string('g4SimHitsMuonME0Hits'),
    digiPreRecoModelString = cms.string('PreRecoGaussian'),
    timeResolution = cms.double(0.0), # in ns
    phiResolution = cms.double(0.0), # in cm average resolution along local x in case of no correlation
    etaResolution = cms.double(0.0), # in cm average resolution along local y in case of no correlation
    phiError = cms.double(0.001), # normally error should be the resolution, but for the case resolution = 0
    etaError = cms.double(0.001), # normally error should be the resolution, but for the case resolution = 0
    constantPhiSpatialResolution = cms.bool(True),
    useCorrelation = cms.bool(False),
    useEtaProjectiveGEO = cms.bool(False),
    averageEfficiency = cms.double(0.98),
    gaussianSmearing = cms.bool(True),          # False --> Uniform smearing
    digitizeOnlyMuons = cms.bool(False),
    # simulateIntrinsicNoise = cms.bool(False), # intrinsic noise --> not implemented
    # averageNoiseRate = cms.double(0.001),     # intrinsic noise --> not implemented
    simulateElectronBkg = cms.bool(True),       # True - will simulate electron background
    simulateNeutralBkg  = cms.bool(True),       # True - will simulate neutral (n+g)  background
    minBunch = cms.int32(-5),                   # [x 25 ns], forms the readout window together with maxBunch,
    maxBunch = cms.int32(3),                    # we should think of shrinking this window ...
    instLumi = cms.double(7.5),                 # in units of 1E34 cm^-2 s^-1. Internally the background is parametrized from FLUKA+GEANT results at 5x10^34 (PU140). We are adding a 1.5 factor for PU200
    rateFact = cms.double(2.0),                 # We are adding also a safety factor of 2 to take into account the new beam pipe effect (not yet known). Hits can be thrown away later at re-digi step.
    mixLabel = cms.string('mix'),
)

# Module to create simulated ME0 Pre Reco digis.
simMuonME0Digis = cms.EDProducer("ME0DigiPreRecoProducer",
    me0PreRecoDigiCommonParameters
)

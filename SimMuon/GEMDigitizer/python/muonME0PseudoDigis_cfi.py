import FWCore.ParameterSet.Config as cms

from SimMuon.GEMDigitizer.muonGEMDigis_cfi import gemDigiCommonParameters

me0PseudoDigiCommonParameters = cms.PSet(
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
    simulateElectronBkg = cms.bool(False),      # True - will simulate electron background
    simulateNeutralBkg  = cms.bool(False),      # True - will simulate neutral (n+g)  background
    minBunch = cms.int32(-5),                   # [x 25 ns], forms the readout window together with maxBunch,
    maxBunch = cms.int32(3),                    # we should think of shrinking this window ...
    instLumi = gemDigiCommonParameters.instLumi,# in units of 1E34 cm^-2 s^-1. Internally the background is parametrized from FLUKA+GEANT results at 5x10^34 (PU140). We are adding a 1.5 factor for PU200
    rateFact = gemDigiCommonParameters.rateFact,# We are adding also a safety factor of 2 to take into account the new beam pipe effect (not yet known). Hits can be thrown away later at re-digi step. Parameters are kept in sync with the ones used in the GEM digitizer
    referenceInstLumi = gemDigiCommonParameters.referenceInstLumi, #reference inst. luminosity 5x10^34 cm-2s-1
    mixLabel = cms.string('mix')
)

from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
premix_stage2.toModify(me0PseudoDigiCommonParameters, mixLabel = "mixData")

# Module to create simulated ME0 Pre Reco digis.
simMuonME0PseudoDigis = cms.EDProducer("ME0DigiPreRecoProducer",
    me0PseudoDigiCommonParameters
)

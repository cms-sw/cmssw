import FWCore.ParameterSet.Config as cms

gemDigiCommonParameters = cms.PSet(
    signalPropagationSpeed = cms.double(0.66),
    cosmics = cms.bool(False),
    timeResolution = cms.double(5),
    timeJitter = cms.double(1.0),
    averageShapingTime = cms.double(50.0),
    averageEfficiency = cms.double(0.98),
    averageNoiseRate = cms.double(0.001), #intrinsic noise
    bxwidth = cms.int32(25),
    minBunch = cms.int32(-5), ## in terms of 25 ns
    maxBunch = cms.int32(3),
    mixLabel = cms.string('mix'),	# added by A.Sharma
    inputCollection = cms.string('g4SimHitsMuonGEMHits'),
    digiModelString = cms.string('Simple'),
    digitizeOnlyMuons = cms.bool(False),
    doBkgNoise = cms.bool(False), #False == No background simulation
    doNoiseCLS = cms.bool(True),
    fixedRollRadius = cms.bool(True), #Uses fixed radius in the center of the roll
    simulateIntrinsicNoise = cms.bool(False),
    simulateElectronBkg = cms.bool(True),	#False=simulate only neutral Bkg
    instLumi = cms.double(7.5), # in units of 1E34 cm^-2 s^-1. Internally the background is parametrized from FLUKA+GEANT results at 5x10^34 (PU140). We are adding a 1.5 factor for PU200
    rateFact = cms.double(1.0), # Set this factor to 1 since the new background model includes the new beam pipe and the relevant effects, so no need of higher safety factor. keeping is here is just for backward compatibiliy
    referenceInstLumi = cms.double(5.) #In units of 10^34 Hz/cm^2. Internally the functions based on the FLUKA+GEANT simulation are normalized to 5x10^34 Hz/cm^2, this is needed to rescale them properly
)

# Module to create simulated GEM digis.
simMuonGEMDigis = cms.EDProducer("GEMDigiProducer",
    gemDigiCommonParameters
)


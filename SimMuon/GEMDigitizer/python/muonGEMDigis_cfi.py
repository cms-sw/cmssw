import FWCore.ParameterSet.Config as cms

gemDigiCommonParameters = cms.PSet(
    signalPropagationSpeed = cms.double(0.66),
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
    referenceInstLumi = cms.double(5.), #In units of 10^34 Hz/cm^2. Internally the functions based on the FLUKA+GEANT simulation are normalized to 5x10^34 Hz/cm^2, this is needed to rescale them properly
    resolutionX = cms.double(0.03), # referenced 2014 Test Beam results.
#the following parameters are needed to model the neutron induced background contribution.
#The parameters have been obtained after the fit of the rates predicted by FLUKA.
#By default the backgroud modeling with these parameters should be disabled with the 92X release setting doBkgNoise=False   
    GE11ElecBkgParam0 = cms.double(406.249),
    GE11ElecBkgParam1 = cms.double(-2.90939),
    GE11ElecBkgParam2 = cms.double(0.00548191),
    GE21ElecBkgParam0 = cms.double(97.0505),
    GE21ElecBkgParam1 = cms.double(-0.452612),
    GE21ElecBkgParam2 = cms.double(0.000550599),
    GE11ModNeuBkgParam0 = cms.double(5710.23),
    GE11ModNeuBkgParam1 = cms.double(-43.3928),
    GE11ModNeuBkgParam2 = cms.double(0.0863681),
    GE21ModNeuBkgParam0 = cms.double(1440.44),
    GE21ModNeuBkgParam1 = cms.double(-7.48607),
    GE21ModNeuBkgParam2 = cms.double(0.0103078)
)

from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
premix_stage2.toModify(gemDigiCommonParameters, mixLabel = "mixData")

# Module to create simulated GEM digis.
simMuonGEMDigis = cms.EDProducer("GEMDigiProducer",
    gemDigiCommonParameters
)


import FWCore.ParameterSet.Config as cms

from SimMuon.GEMDigitizer.muonGEMDigis_cfi import gemDigiCommonParameters

me0DigiCommonParameters = cms.PSet(
    signalPropagationSpeed = cms.double(0.66),
    cosmics = cms.bool(False),
    timeResolution = cms.double(5),
    timeJitter = cms.double(1.0),
    averageShapingTime = cms.double(50.0),
    #clsParametrization = cms.vdouble(0.455091, 0.865613, 0.945891, 0.973286, 0.986234, 0.991686, 0.996865, 0.998501, 1.),
    averageEfficiency = cms.double(0.98),
    averageNoiseRate = cms.double(0.001), #intrinsic noise
    bxwidth = cms.int32(25),
    minBunch = cms.int32(-5), ## in terms of 25 ns
    maxBunch = cms.int32(3),
    mixLabel = cms.string('mix'),	# added by A.Sharma
    inputCollection = cms.string('g4SimHitsMuonME0Hits'),
    digiModelString = cms.string('Simple'),
    digitizeOnlyMuons = cms.bool(False),
    doBkgNoise = cms.bool(False), #False == No background simulation
    doNoiseCLS = cms.bool(True),
    fixedRollRadius = cms.bool(True), #Uses fixed radius in the center of the roll
    simulateIntrinsicNoise = cms.bool(False),
    simulateElectronBkg = cms.bool(True),	#False=simulate only neutral Bkg
    instLumi = gemDigiCommonParameters.instLumi, # in units of 1E34 cm^-2 s^-1. Internally the background is parametrized from FLUKA+GEANT results at 5x10^34 (PU140). We are adding a 1.5 factor for PU200
    rateFact = gemDigiCommonParameters.rateFact, # We are adding also a safety factor of 2 to take into account the new beam pipe effect (not yet known). Hits can be thrown away later at re-digi step. Parameters are kept in sync with the ones used in the GEM digitizer
    referenceInstLumi = gemDigiCommonParameters.referenceInstLumi  #reference inst. luminosity 5x10^34 cm-2s-1

)

# Module to create simulated ME0 digis.
simMuonME0Digis = cms.EDProducer("ME0DigiProducer",
    me0DigiCommonParameters
)


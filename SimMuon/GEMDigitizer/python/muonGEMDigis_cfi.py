import FWCore.ParameterSet.Config as cms

# Module to create simulated GEM digis.
simMuonGEMDigis = cms.EDProducer("GEMDigiProducer",
    signalPropagationSpeed = cms.double(0.66),
    cosmics = cms.bool(False),
    timeResolution = cms.double(5),
    timeJitter = cms.double(1.0),
    averageShapingTime = cms.double(50.0),
#    averageClusterSize = cms.double(1.5),#1.5
#cumulative cls distribution at 771 mkA; experimental results from test pion beam
    clsParametrization = cms.vdouble(0.455091, 0.865613, 0.945891, 0.973286, 0.986234, 0.991686, 0.996865, 0.998501, 1.),
#cumulative cls distribution at 752 mkA; experimental results from test pion beam
#    clsParametrization = cms.vdouble(0.663634, 0.928571, 0.967544, 0.982665, 0.990288, 0.994222, 0.997541, 0.999016, 1.),
    averageEfficiency = cms.double(0.98),
    averageNoiseRate = cms.double(0.001), #intrinsic noise
    bxwidth = cms.int32(25),
    minBunch = cms.int32(-5), ## in terms of 25 ns
    maxBunch = cms.int32(3),
    inputCollection = cms.string('g4SimHitsMuonGEMHits'),
    digiModelString = cms.string('Simple'),
    digitizeOnlyMuons = cms.bool(True),
    doBkgNoise = cms.bool(True), #False == No background simulation
    doNoiseCLS = cms.bool(True),
    fixedRollRadius = cms.bool(True), #Uses fixed radius in the center of the roll
    minPabsNoiseCLS = cms.double(0.),
    simulateIntrinsicNoise = cms.bool(False),
    scaleLumi = cms.double(1.),
#Parameters for background model
    simulateElectronBkg = cms.bool(False),	#False=simulate only neutral Bkg
#const and slope for the expo model of neutral bkg for GE1/1:
    constNeuGE11 = cms.double(807.1),
    slopeNeuGE11 = cms.double(-0.01443),
#params for the simple pol5 model of neutral bkg for GE2/1:
    GE21NeuBkgParams = cms.vdouble(2954.04, -58.7558, 0.473481, -0.00188292, 3.67041e-06, -2.80261e-09),
#params for the simple pol3 model of electron bkg for GE1/1:
    GE11ElecBkgParams = cms.vdouble(2135.93, -33.1251, 0.177738, -0.000319334),
#params for the simple pol6 model of electron bkg for GE2/1:
    GE21ElecBkgParams = cms.vdouble(-43642.2, 1335.98, -16.476, 0.105281, -0.000368758, 6.72937e-07, -5.00872e-10)

)


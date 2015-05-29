import FWCore.ParameterSet.Config as cms

# Module to create simulated ME0 Pre Reco digis.
simMuonME0Digis = cms.EDProducer("ME0DigiPreRecoProducer",
    mixLabel = cms.string('mix'),
    inputCollection = cms.string('g4SimHitsMuonME0Hits'),
    digiPreRecoModelString = cms.string('PreRecoGaussian'),
    timeResolution = cms.double(0.010),      # [in ns] => for now at 10ps 
    phiResolution = cms.double(0.05),        # [in cm] average resolution along local x in case of no correlation
    etaResolution = cms.double(1.),          # [in cm] average resolution along local y in case of no correlation
    useCorrelation  = cms.bool(False),
    useEtaProjectiveGEO  = cms.bool(False),
    averageEfficiency = cms.double(0.98),
    doBkgNoise = cms.bool(False),            # False => No background noise simulation
    digitizeOnlyMuons = cms.bool(False),
    simulateIntrinsicNoise = cms.bool(False),
    simulateElectronBkg = cms.bool(False),   # True => will simulate electron background
    averageNoiseRate = cms.double(0.001),    # simulation of intrinsic noise
    bxwidth = cms.int32(25),
    minBunch = cms.int32(-5),                # in bx-units (x 25 ns)
    maxBunch = cms.int32(3)
)

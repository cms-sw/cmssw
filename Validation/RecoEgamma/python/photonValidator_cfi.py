import FWCore.ParameterSet.Config as cms

import SimTracker.TrackAssociation.TrackAssociatorByHits_cfi
trackAssociatorByHitsForPhotonValidation = SimTracker.TrackAssociation.TrackAssociatorByHits_cfi.TrackAssociatorByHits.clone()
trackAssociatorByHitsForPhotonValidation.ComponentName = cms.string('trackAssociatorByHitsForPhotonValidation')
trackAssociatorByHitsForPhotonValidation.Cut_RecoToSim = 0.5
trackAssociatorByHitsForPhotonValidation.Quality_SimToReco = 0.5
trackAssociatorByHitsForPhotonValidation.Purity_SimToReco = 0.5
trackAssociatorByHitsForPhotonValidation.SimToRecoDenominator = 'reco'


photonValidation = cms.EDAnalyzer("PhotonValidator",
    ComponentName = cms.string('photonValidation'),
    OutputFileName = cms.string('PhotonValidationHistos.root'),
    scEndcapProducer = cms.string('correctedMulti5x5SuperClustersWithPreshower'),
    scBarrelProducer = cms.string('correctedHybridSuperClusters'),
    phoProducer = cms.string('photons'),
    pfCandidates = cms.InputTag("particleFlow"),
    #valueMapPhoToParticleBasedIso = cms.InputTag("particleBasedIsolation","valMapPhoToPFisolation"),
    valueMapPhoToParticleBasedIso = cms.string("gedPhotons"),                                                            
    conversionOITrackProducer =cms.string('ckfOutInTracksFromConversions'),
    conversionIOTrackProducer =cms.string('ckfInOutTracksFromConversions'),
    outInTrackCollection =cms.string(''),
    inOutTrackCollection =cms.string(''),
    photonCollection = cms.string(''),
    hbheInstance = cms.string(''),
    endcapEcalHits = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    barrelEcalHits = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    hbheModule = cms.string('hbhereco'),
    trackProducer = cms.InputTag("generalTracks"),
    label_tp = cms.InputTag("tpSelection"),
    Verbosity = cms.untracked.int32(0),
    fastSim = cms.bool(False),
    isRunCentrally = cms.bool(False),
    analyzerName = cms.string('PhotonValidator'),
#
    minPhoEtCut = cms.double(10.),
    convTrackMinPtCut = cms.double(1.),
    likelihoodCut = cms.double(0.),
#
    useTP = cms.bool(True),
#
    eBin = cms.int32(100),
    eMin = cms.double(0.0),
    eMax = cms.double(500.0),
#
    etScale = cms.double(0.1),
#
    etBin = cms.int32(100),
    etMax = cms.double(250.),
    etMin = cms.double(0.0),
#
    etaBin = cms.int32(100),
    etaBin2 = cms.int32(25),
    etaMin = cms.double(-2.5),
    etaMax = cms.double(2.5),
#
    phiBin = cms.int32(100),
    phiMin = cms.double(-3.14),
    phiMax = cms.double(3.14),
#
    r9Bin = cms.int32(200),
    r9Min = cms.double(0.0),
    r9Max = cms.double(1.1),
#
    dEtaTracksBin = cms.int32(100),
    dEtaTracksMin = cms.double(-0.2),
    dEtaTracksMax = cms.double(0.2),
#
    dPhiTracksBin = cms.int32(100),
    dPhiTracksMin = cms.double(-0.5),
    dPhiTracksMax = cms.double(0.5),
#
    dEtaBin = cms.int32(100),
    dEtaMin = cms.double(-0.2),
    dEtaMax = cms.double(0.2),
#
    dPhiBin = cms.int32(100),
    dPhiMin = cms.double(-0.05),
    dPhiMax = cms.double(0.05),
#
    rBin = cms.int32(48),
    rMin = cms.double(0.),
    rMax = cms.double(120),
#
    zBin = cms.int32(100),
    zMin = cms.double(-220.),
    zMax = cms.double(220),
#
    resBin = cms.int32(100),
    resMin = cms.double(0.7),
    resMax = cms.double(1.2),
#
    dCotCutOn =cms.bool(False),
    dCotCutValue=cms.double(0.05),
    dCotHardCutValue=cms.double(0.02),
#
    dCotTracksBin = cms.int32(100),
    dCotTracksMin = cms.double(-2.),
    dCotTracksMax = cms.double(2.),
#
    povereBin = cms.int32(100),
    povereMin = cms.double(0.),
    povereMax = cms.double(5.),
#
    eoverpBin = cms.int32(100),
    eoverpMin = cms.double(0.),
    eoverpMax = cms.double(5.),
#
    chi2Min = cms.double(0.),
    chi2Max = cms.double(20.),
#
    ggMassBin =cms.int32(200),
    ggMassMin =cms.double(60.),
    ggMassMax =cms.double(160.),
#
    rBinForXray = cms.int32(200),
    rMinForXray = cms.double(0.),
    rMaxForXray = cms.double(80.),
    zBinForXray = cms.int32(100),
    zBin2ForXray = cms.int32(560),
    zMinForXray = cms.double(0.),
    zMaxForXray = cms.double(280.),
                                  
# Unused stuff
    hcalIsolExtR = cms.double(0.3),
    hcalIsolInnR = cms.double(0.0),
    minTrackPtCut = cms.double(1.5),
    minBcEtCut = cms.double(0.0),
    lipCut = cms.double(2.0),
    trkIsolInnR = cms.double(0.03),
    ecalIsolR = cms.double(0.35),
    trkIsolExtR = cms.double(0.3),
    maxNumOfTracksInCone = cms.int32(3),
    hcalEtSumCut = cms.double(6.0),
    minHcalHitEtCut = cms.double(0.0),

    trkPtSumCut = cms.double(9999.0),
    ecalEtSumCut = cms.double(5.0),
 
)



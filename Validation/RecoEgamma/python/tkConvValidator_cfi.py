import FWCore.ParameterSet.Config as cms
from Validation.RecoEgamma.tpSelection_cfi import *

from SimTracker.TrackAssociation.quickTrackAssociatorByHits_cfi import *
import SimTracker.TrackAssociation.quickTrackAssociatorByHits_cfi
trackAssociatorByHitsForConversionValidation = SimTracker.TrackAssociation.quickTrackAssociatorByHits_cfi.quickTrackAssociatorByHits.clone()
trackAssociatorByHitsForConversionValidation.ComponentName = cms.string('trackAssociatorByHitsForConversionValidation')
trackAssociatorByHitsForConversionValidation.SimToRecoDenominator = 'reco'
trackAssociatorByHitsForConversionValidation.Quality_SimToReco = 0.5
trackAssociatorByHitsForConversionValidation.Purity_SimToReco = 0.5
trackAssociatorByHitsForConversionValidation.Cut_RecoToSim = 0.5

import PhysicsTools.RecoAlgos.trackingParticleSelector_cfi
tpSelecForEfficiency = PhysicsTools.RecoAlgos.trackingParticleSelector_cfi.trackingParticleSelector.clone()
tpSelecForEfficiency.chargedOnly = True
# trackingParticleSelector.pdgId = cms.vint32()
tpSelecForEfficiency.tip = 120
tpSelecForEfficiency.lip = 280
tpSelecForEfficiency.signalOnly = False
tpSelecForEfficiency.minRapidity = -2.5
tpSelecForEfficiency.ptMin = 0.3
tpSelecForEfficiency.maxRapidity = 2.5
tpSelecForEfficiency.minHit = 0


tpSelecForFakeRate = PhysicsTools.RecoAlgos.trackingParticleSelector_cfi.trackingParticleSelector.clone()
tpSelecForFakeRate.chargedOnly = True
# trackingParticleSelector.pdgId = cms.vint32()
tpSelecForFakeRate.tip = 120
tpSelecForFakeRate.lip = 280
tpSelecForFakeRate.signalOnly = False
tpSelecForFakeRate.minRapidity = -2.5
tpSelecForFakeRate.ptMin = 0.
tpSelecForFakeRate.maxRapidity = 2.5
tpSelecForFakeRate.minHit = 0



tkConversionValidation = cms.EDAnalyzer("TkConvValidator",
    Name = cms.untracked.string('tkConversionValidation'),
    isRunCentrally = cms.bool(False),
    OutputFileName = cms.string('ValidationHistos.root'),
    convProducer = cms.string('allConversions'),
    conversionCollection = cms.string(''),
    phoProducer = cms.string('photons'),
    photonCollection = cms.string(''),                                    
    trackProducer = cms.InputTag("generalTracks"),
    dqmpath = cms.string('EgammaV/ConversionValidator/'),
    Verbosity = cms.untracked.int32(0),
    generalTracksOnly = cms.bool(True),
    arbitratedMerged =  cms.bool(False),
    arbitratedEcalSeeded = cms.bool(False),
    ecalalgotracks = cms.bool(False),
    highPurity = cms.bool(True),
    minProb = cms.double(-99.9),
    maxHitsBeforeVtx = cms.uint32(999),
    minLxy = cms.double(-9999.9),

    minPhoPtForEffic = cms.double(0.3),#when hardcoded it was 2.5
    maxPhoEtaForEffic = cms.double(2.5),
    maxPhoZForEffic = cms.double(200.),
    maxPhoRForEffic = cms.double(100.),
    minPhoPtForPurity = cms.double(0.),#when hardcoded it was 0.5
    maxPhoEtaForPurity = cms.double(2.5),
    maxPhoZForPurity = cms.double(200.),
    maxPhoRForPurity = cms.double(100.),

#
    minPhoEtCut = cms.double(0.),

#
    useTP =  cms.bool(True),
#

    etBin = cms.int32(100),                                  
    etMax = cms.double(100.0),                                  
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
    resBin = cms.int32(100),
    resMin = cms.double(0.),
    resMax = cms.double(1.1),
#
    eoverpBin =  cms.int32(100),
    eoverpMin =  cms.double(0.),
    eoverpMax =  cms.double(5.),
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
    rBin = cms.int32(60), 
    rMin = cms.double(0.),
    rMax = cms.double(120),
#
    zBin = cms.int32(100),
    zMin = cms.double(-220.),
    zMax = cms.double(220),
#
 
    dCotTracksBin = cms.int32(100),                              
    dCotTracksMin = cms.double(-0.12),
    dCotTracksMax = cms.double(0.12),
#                                  
    chi2Min =  cms.double(0.),
    chi2Max =  cms.double(20.),                              
#

    rBinForXray = cms.int32(200),
    rMinForXray = cms.double(0.),
    rMaxForXray = cms.double(80.),                               
    zBinForXray = cms.int32(100),
    zBin2ForXray = cms.int32(560),
    zMinForXray = cms.double(0.),
    zMaxForXray = cms.double(280.),                               
                                  
)



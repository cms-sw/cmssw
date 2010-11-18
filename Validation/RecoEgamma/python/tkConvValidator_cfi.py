import FWCore.ParameterSet.Config as cms
from Validation.RecoEgamma.tpSelection_cfi import *
from SimTracker.TrackAssociation.TrackAssociatorByHits_cfi import *
import SimTracker.TrackAssociation.TrackAssociatorByHits_cfi
TrackAssociatorByHits.SimToRecoDenominator = 'reco'
TrackAssociatorByHits.Quality_SimToReco = 0.5
TrackAssociatorByHits.Purity_SimToReco = 0.5
TrackAssociatorByHits.Cut_RecoToSim = 0.5

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
    convProducer = cms.string('trackerOnlyConversions'),
    conversionCollection = cms.string(''),
    trackProducer = cms.InputTag("generalTracks"),
    Verbosity = cms.untracked.int32(0),
    mergedTracks =  cms.bool(False),

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
    rBin = cms.int32(40), 
    rMin = cms.double(0.),
    rMax = cms.double(80),
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
    zMaxForXray = cms.double(280.)                               
                                  
)



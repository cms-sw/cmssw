import FWCore.ParameterSet.Config as cms
import math

eleConvRejectionValidation = cms.EDAnalyzer("ElectronConversionRejectionValidator",
    Name = cms.untracked.string('eleConvRejectionValidation'),
    isRunCentrally = cms.bool(False),
    OutputFileName = cms.string('ValidationHistos.root'),
    convProducer = cms.string('allConversions'),
    conversionCollection = cms.string(''),
    gsfElectronProducer = cms.string('gsfElectrons'),
    gsfElectronCollection = cms.string(''),                                    
    dqmpath = cms.string('EgammaV/ElectronConversionRejectionValidator/'),
    Verbosity = cms.untracked.int32(0),

    elePtMin = cms.double(10.0),
    eleExpectedHitsInnerMax = cms.int32(0),
    eleD0Max = cms.double(0.02),

    ptBin = cms.int32(100),                                  
    ptMax = cms.double(100.0),                                  
    ptMin = cms.double(0.0),
    
    trackptBin = cms.int32(100),                                  
    trackptMax = cms.double(100.0),                                  
    trackptMin = cms.double(0.0),    
#
    etaBin = cms.int32(100),
    etaMin = cms.double(-2.5),
    etaMax = cms.double(2.5),
#
    phiBin = cms.int32(100),
#
    #resBin = cms.int32(100),
    #resMin = cms.double(0.),
    #resMax = cms.double(1.1),
#
    #eoverpBin =  cms.int32(100),
    #eoverpMin =  cms.double(0.),
    #eoverpMax =  cms.double(5.),
#                                       
    #dEtaTracksBin = cms.int32(100),
    #dEtaTracksMin = cms.double(-0.2),
    #dEtaTracksMax = cms.double(0.2),
#
    #dPhiTracksBin = cms.int32(100),
    #dPhiTracksMin = cms.double(-0.5),
    #dPhiTracksMax = cms.double(0.5),
#
    #dEtaBin = cms.int32(100),
    #dEtaMin = cms.double(-0.2),
    #dEtaMax = cms.double(0.2),
#
    #dPhiBin = cms.int32(100),
    #dPhiMin = cms.double(-0.05),
    #dPhiMax = cms.double(0.05),
#
    rhoBin = cms.int32(120), 
    rhoMin = cms.double(0.),
    rhoMax = cms.double(120),
#
    zBin = cms.int32(100),
    zMin = cms.double(-220.),
    zMax = cms.double(220),
#
 
    #dCotTracksBin = cms.int32(100),                              
    #dCotTracksMin = cms.double(-0.12),
    #dCotTracksMax = cms.double(0.12),
#                                  
                         
                                  
)



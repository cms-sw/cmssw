import FWCore.ParameterSet.Config as cms

from Validation.RecoEgamma.photonValidator_cfi import *


photonPostprocessing = cms.EDAnalyzer("PhotonPostprocessing",

    Name = cms.untracked.string('photonPostprocessing'),

    standAlone = cms.bool(False),
    batch = cms.bool(False),                                     


    etBin = photonValidation.etBin,
    etMin = photonValidation.etMin,
    etMax = photonValidation.etMax,

    eBin = photonValidation.etBin,
    eMin = photonValidation.etMin,
    eMax = photonValidation.etMax,

    etaBin = photonValidation.etaBin,
    etaBin2 = photonValidation.etaBin2,
    etaMin = photonValidation.etaMin,
    etaMax = photonValidation.etaMax,

    phiBin = photonValidation.phiBin,
    phiMin = photonValidation.phiMin,
    phiMax = photonValidation.phiMax,

    rBin = photonValidation.rBin,
    rMin = photonValidation.rMin,
    rMax = photonValidation.rMax,

    zBin = photonValidation.zBin,
    zMin = photonValidation.zMin,
    zMax = photonValidation.zMax,

                                      
                                     
    InputFileName = cms.string("inputPhotonValidation.root"),
                                     
    OutputFileName = cms.string('standaloneOutputValidation.root'),
)

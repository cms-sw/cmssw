import FWCore.ParameterSet.Config as cms

from Validation.RecoEgamma.tkConvValidator_cfi import *


conversionPostprocessing = cms.EDAnalyzer("ConversionPostprocessing",

    Name = cms.untracked.string('conversionPostprocessing'),

    standAlone = cms.bool(False),
    batch = cms.bool(False),                                     


    etBin = tkConversionValidation.etBin,
    etMin = tkConversionValidation.etMin,
    etMax = tkConversionValidation.etMax,

    etaBin = tkConversionValidation.etaBin,
    etaBin2 = tkConversionValidation.etaBin2,
    etaMin = tkConversionValidation.etaMin,
    etaMax = tkConversionValidation.etaMax,

    phiBin = tkConversionValidation.phiBin,
    phiMin = tkConversionValidation.phiMin,
    phiMax = tkConversionValidation.phiMax,

    rBin = tkConversionValidation.rBin,
    rMin = tkConversionValidation.rMin,
    rMax = tkConversionValidation.rMax,

    zBin = tkConversionValidation.zBin,
    zMin = tkConversionValidation.zMin,
    zMax = tkConversionValidation.zMax,

                                      
                                     
    InputFileName = cms.string("inputPhotonValidation.root"),
                                     
    OutputFileName = cms.string('standaloneOutputValidation.root'),
)

import FWCore.ParameterSet.Config as cms
from Validation.RecoTau.dataTypes.ValidateTausOnRealData_cff import *
from Validation.RecoTau.dataTypes.ValidateTausOnRealElectronsData_cff import *
from Validation.RecoTau.dataTypes.ValidateTausOnRealMuonsData_cff import *
from Validation.RecoTau.dataTypes.ValidateTausOnZEE_cff import *
from Validation.RecoTau.dataTypes.ValidateTausOnZMM_cff import *
from Validation.RecoTau.dataTypes.ValidateTausOnZTT_cff import *
from Validation.RecoTau.dataTypes.ValidateTausOnQCD_cff import *

from Validation.RecoTau.RecoTauValidationMiniAOD_cfi import *
tauValidationMiniAODZTT = tauValidationMiniAOD.clone()
discs_to_retain = ['decayModeFinding', 'CombinedIsolationDeltaBetaCorr3HitsdR03', 'IsolationMVArun2v1DBoldDMwLT', 'IsolationMVArun2v1DBnewDMwLT', 'againstMuon', 'againstElectron']
tauValidationMiniAODZTT.discriminators = cms.VPSet([p for p in tauValidationMiniAODZTT.discriminators if any(disc in p.discriminator.value() for disc in discs_to_retain) ])

tauValidationMiniAODZEE = tauValidationMiniAODZTT.clone(
  RefCollection = cms.InputTag("kinematicSelectedTauValDenominatorZEE"),
  ExtensionName = cms.string('ZEE')
)
tauValidationMiniAODZMM = tauValidationMiniAODZTT.clone(
  RefCollection = cms.InputTag("kinematicSelectedTauValDenominatorZMM"),
  ExtensionName = cms.string('ZMM')
)
tauValidationMiniAODQCD = tauValidationMiniAODZTT.clone(
  RefCollection = cms.InputTag("kinematicSelectedTauValDenominatorQCD"),
  ExtensionName = cms.string('QCD')
)
tauValidationMiniAODRealData = tauValidationMiniAODZTT.clone(
  RefCollection = cms.InputTag("CleanedPFJets"),
  ExtensionName = cms.string('JETHT')
)
tauValidationMiniAODRealElectronsData = tauValidationMiniAODZTT.clone(
  RefCollection = cms.InputTag("ElZLegs","theProbeLeg"),
  ExtensionName = cms.string("DoubleElectron")
)
tauValidationMiniAODRealMuonsData = tauValidationMiniAODZTT.clone(
  RefCollection = cms.InputTag("MuZLegs","theProbeLeg"),
  ExtensionName = cms.string('DoubleMuon')
)


from DQMServices.Core.DQMEDHarvester import DQMEDHarvester
efficienciesTauValidationMiniAODZTT = cms.EDProducer("TauDQMHistEffProducer",
    plots = cms.PSet(
        Summary = cms.PSet(
            denominator = cms.string('RecoTauV/miniAODValidation/ZTT/#PAR#PlotDen'),
            efficiency = cms.string('RecoTauV/miniAODValidation/ZTT/#PAR#Plot'),
            numerator = cms.string('RecoTauV/miniAODValidation/ZTT/#PAR#PlotNum'),
            parameter = cms.vstring('summary'),
            stepByStep = cms.bool(True)
        ),
    )
)
efficienciesTauValidationMiniAODZEE = cms.EDProducer("TauDQMHistEffProducer",
    plots = cms.PSet(
        Summary = cms.PSet(
            denominator = cms.string('RecoTauV/miniAODValidation/ZEE/#PAR#PlotDen'),
            efficiency = cms.string('RecoTauV/miniAODValidation/ZEE/#PAR#Plot'),
            numerator = cms.string('RecoTauV/miniAODValidation/ZEE/#PAR#PlotNum'),
            parameter = cms.vstring('summary'),
            stepByStep = cms.bool(True)
        ),
    )
)
efficienciesTauValidationMiniAODZMM = cms.EDProducer("TauDQMHistEffProducer",
    plots = cms.PSet(
        Summary = cms.PSet(
            denominator = cms.string('RecoTauV/miniAODValidation/ZMM/#PAR#PlotDen'),
            efficiency = cms.string('RecoTauV/miniAODValidation/ZMM/#PAR#Plot'),
            numerator = cms.string('RecoTauV/miniAODValidation/ZMM/#PAR#PlotNum'),
            parameter = cms.vstring('summary'),
            stepByStep = cms.bool(True)
        ),
    )
)
efficienciesTauValidationMiniAODQCD = cms.EDProducer("TauDQMHistEffProducer",
    plots = cms.PSet(
        Summary = cms.PSet(
            denominator = cms.string('RecoTauV/miniAODValidation/QCD/#PAR#PlotDen'),
            efficiency = cms.string('RecoTauV/miniAODValidation/QCD/#PAR#Plot'),
            numerator = cms.string('RecoTauV/miniAODValidation/QCD/#PAR#PlotNum'),
            parameter = cms.vstring('summary'),
            stepByStep = cms.bool(True)
        ),
    )
)
efficienciesTauValidationMiniAODRealData = cms.EDProducer("TauDQMHistEffProducer",
    plots = cms.PSet(
        Summary = cms.PSet(
            denominator = cms.string('RecoTauV/miniAODValidation/JETHT/#PAR#PlotDen'),
            efficiency = cms.string('RecoTauV/miniAODValidation/JETHT/#PAR#Plot'),
            numerator = cms.string('RecoTauV/miniAODValidation/JETHT/#PAR#PlotNum'),
            parameter = cms.vstring('summary'),
            stepByStep = cms.bool(True)
        ),
    )
)
efficienciesTauValidationMiniAODRealElectronsData = cms.EDProducer("TauDQMHistEffProducer",
    plots = cms.PSet(
        Summary = cms.PSet(
            denominator = cms.string('RecoTauV/miniAODValidation/DoubleElectron/#PAR#PlotDen'),
            efficiency = cms.string('RecoTauV/miniAODValidation/DoubleElectron/#PAR#Plot'),
            numerator = cms.string('RecoTauV/miniAODValidation/DoubleElectron/#PAR#PlotNum'),
            parameter = cms.vstring('summary'),
            stepByStep = cms.bool(True)
        ),
    )
)
efficienciesTauValidationMiniAODRealMuonsData = cms.EDProducer("TauDQMHistEffProducer",
    plots = cms.PSet(
        Summary = cms.PSet(
            denominator = cms.string('RecoTauV/miniAODValidation/DoubleMuon/#PAR#PlotDen'),
            efficiency = cms.string('RecoTauV/miniAODValidation/DoubleMuon/#PAR#Plot'),
            numerator = cms.string('RecoTauV/miniAODValidation/DoubleMuon/#PAR#PlotNum'),
            parameter = cms.vstring('summary'),
            stepByStep = cms.bool(True)
        ),
    )
)

tauValidationSequenceMiniAOD = cms.Sequence(produceDenominatorZTT*tauValidationMiniAODZTT*produceDenominatorZEE*tauValidationMiniAODZEE*produceDenominatorZMM*tauValidationMiniAODZMM*produceDenominatorQCD*tauValidationMiniAODQCD*tauValidationMiniAODRealData*tauValidationMiniAODRealElectronsData*tauValidationMiniAODRealMuonsData)

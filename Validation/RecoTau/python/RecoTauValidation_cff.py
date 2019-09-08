import FWCore.ParameterSet.Config as cms

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
  ExtensionName = cms.string('RealData')
)
tauValidationMiniAODRealElectronsData = tauValidationMiniAODZTT.clone(
  RefCollection = cms.InputTag("ElZLegs","theProbeLeg"),
  ExtensionName = cms.string("RealElectronsData")
)
tauValidationMiniAODRealMuonsData = tauValidationMiniAODZTT.clone(
  RefCollection = cms.InputTag("MuZLegs","theProbeLeg"),
  ExtensionName = cms.string('RealMuonsData')
)


from DQMServices.Core.DQMEDHarvester import DQMEDHarvester
efficienciesTauValidationMiniAODZTT = cms.EDProducer("TauDQMHistEffProducer",
    plots = cms.PSet(
        Summary = cms.PSet(
            denominator = cms.string('RecoTauV/miniAODValidationZTT/#PAR#PlotDen'),
            efficiency = cms.string('RecoTauV/miniAODValidationZTT/#PAR#Plot'),
            numerator = cms.string('RecoTauV/miniAODValidationZTT/#PAR#PlotNum'),
            parameter = cms.vstring('summary'),
            stepByStep = cms.bool(True)
        ),
    )
)
efficienciesTauValidationMiniAODZEE = cms.EDProducer("TauDQMHistEffProducer",
    plots = cms.PSet(
        Summary = cms.PSet(
            denominator = cms.string('RecoTauV/miniAODValidationZEE/#PAR#PlotDen'),
            efficiency = cms.string('RecoTauV/miniAODValidationZEE/#PAR#Plot'),
            numerator = cms.string('RecoTauV/miniAODValidationZEE/#PAR#PlotNum'),
            parameter = cms.vstring('summary'),
            stepByStep = cms.bool(True)
        ),
    )
)
efficienciesTauValidationMiniAODZMM = cms.EDProducer("TauDQMHistEffProducer",
    plots = cms.PSet(
        Summary = cms.PSet(
            denominator = cms.string('RecoTauV/miniAODValidationZMM/#PAR#PlotDen'),
            efficiency = cms.string('RecoTauV/miniAODValidationZMM/#PAR#Plot'),
            numerator = cms.string('RecoTauV/miniAODValidationZMM/#PAR#PlotNum'),
            parameter = cms.vstring('summary'),
            stepByStep = cms.bool(True)
        ),
    )
)
efficienciesTauValidationMiniAODQCD = cms.EDProducer("TauDQMHistEffProducer",
    plots = cms.PSet(
        Summary = cms.PSet(
            denominator = cms.string('RecoTauV/miniAODValidationQCD/#PAR#PlotDen'),
            efficiency = cms.string('RecoTauV/miniAODValidationQCD/#PAR#Plot'),
            numerator = cms.string('RecoTauV/miniAODValidationQCD/#PAR#PlotNum'),
            parameter = cms.vstring('summary'),
            stepByStep = cms.bool(True)
        ),
    )
)
efficienciesTauValidationMiniAODRealData = cms.EDProducer("TauDQMHistEffProducer",
    plots = cms.PSet(
        Summary = cms.PSet(
            denominator = cms.string('RecoTauV/miniAODValidationRealData/#PAR#PlotDen'),
            efficiency = cms.string('RecoTauV/miniAODValidationRealData/#PAR#Plot'),
            numerator = cms.string('RecoTauV/miniAODValidationRealData/#PAR#PlotNum'),
            parameter = cms.vstring('summary'),
            stepByStep = cms.bool(True)
        ),
    )
)
efficienciesTauValidationMiniAODRealElectronsData = cms.EDProducer("TauDQMHistEffProducer",
    plots = cms.PSet(
        Summary = cms.PSet(
            denominator = cms.string('RecoTauV/miniAODValidationRealElectronsData/#PAR#PlotDen'),
            efficiency = cms.string('RecoTauV/miniAODValidationRealElectronsData/#PAR#Plot'),
            numerator = cms.string('RecoTauV/miniAODValidationRealElectronsData/#PAR#PlotNum'),
            parameter = cms.vstring('summary'),
            stepByStep = cms.bool(True)
        ),
    )
)
efficienciesTauValidationMiniAODRealMuonsData = cms.EDProducer("TauDQMHistEffProducer",
    plots = cms.PSet(
        Summary = cms.PSet(
            denominator = cms.string('RecoTauV/miniAODValidationRealMuonsData/#PAR#PlotDen'),
            efficiency = cms.string('RecoTauV/miniAODValidationRealMuonsData/#PAR#Plot'),
            numerator = cms.string('RecoTauV/miniAODValidationRealMuonsData/#PAR#PlotNum'),
            parameter = cms.vstring('summary'),
            stepByStep = cms.bool(True)
        ),
    )
)

tauValidationSequenceMiniAOD = cms.Sequence(tauValidationMiniAODZTT*tauValidationMiniAODZEE*tauValidationMiniAODZMM*tauValidationMiniAODQCD*tauValidationMiniAODRealData*tauValidationMiniAODRealElectronsData*tauValidationMiniAODRealMuonsData)

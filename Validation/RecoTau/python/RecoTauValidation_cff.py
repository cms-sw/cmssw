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
discs_to_retain = ['decayModeFinding','decayModeFindingNewDMs',
                   'CombinedIsolationDeltaBetaCorr3HitsdR03',
                   'byLooseDeepTau2018v2p5VSjet','byTightDeepTau2018v2p5VSjet',
                   'byLooseDeepTau2018v2p5VSe','byTightDeepTau2018v2p5VSe',
                   'byLooseDeepTau2018v2p5VSmu','byTightDeepTau2018v2p5VSmu']

tauValidationMiniAODZTT.discriminators = cms.VPSet([p for p in tauValidationMiniAODZTT.discriminators if any(disc in p.discriminator.value() for disc in discs_to_retain) ])

tauValidationMiniAODZEE = tauValidationMiniAODZTT.clone(
    RefCollection = "kinematicSelectedTauValDenominatorZEE",
    ExtensionName = 'ZEE'
)
tauValidationMiniAODZMM = tauValidationMiniAODZTT.clone(
    RefCollection = "kinematicSelectedTauValDenominatorZMM",
    ExtensionName = 'ZMM'
)
tauValidationMiniAODQCD = tauValidationMiniAODZTT.clone(
    RefCollection = "kinematicSelectedTauValDenominatorQCD",
    ExtensionName = 'QCD'
)
tauValidationMiniAODRealData = tauValidationMiniAODZTT.clone(
    RefCollection = "CleanedPFJets",
    ExtensionName = 'JETHT'
)
tauValidationMiniAODRealElectronsData = tauValidationMiniAODZTT.clone(
    RefCollection = "ElZLegs:theProbeLeg",
    ExtensionName = 'DoubleElectron'
)
tauValidationMiniAODRealMuonsData = tauValidationMiniAODZTT.clone(
    RefCollection = "MuZLegs:theProbeLeg",
    ExtensionName = 'DoubleMuon'
)


from DQMServices.Core.DQMEDHarvester import DQMEDHarvester
efficienciesTauValidationMiniAODZTT = cms.EDProducer("TauDQMHistEffProducer",
    plots = cms.PSet(
        Summary = cms.PSet(
            denominator = cms.string('RecoTauV/miniAODValidation/ZTT/Summary/#PAR#PlotDen'),
            efficiency = cms.string('RecoTauV/miniAODValidation/ZTT/Summary/#PAR#Plot'),
            numerator = cms.string('RecoTauV/miniAODValidation/ZTT/Summary/#PAR#PlotNum'),
            parameter = cms.vstring('summary'),
            stepByStep = cms.bool(True)
        ),
    )
)
efficienciesTauValidationMiniAODZEE = cms.EDProducer("TauDQMHistEffProducer",
    plots = cms.PSet(
        Summary = cms.PSet(
            denominator = cms.string('RecoTauV/miniAODValidation/ZEE/Summary/#PAR#PlotDen'),
            efficiency = cms.string('RecoTauV/miniAODValidation/ZEE/Summary/#PAR#Plot'),
            numerator = cms.string('RecoTauV/miniAODValidation/ZEE/Summary/#PAR#PlotNum'),
            parameter = cms.vstring('summary'),
            stepByStep = cms.bool(True)
        ),
    )
)
efficienciesTauValidationMiniAODZMM = cms.EDProducer("TauDQMHistEffProducer",
    plots = cms.PSet(
        Summary = cms.PSet(
            denominator = cms.string('RecoTauV/miniAODValidation/ZMM/Summary/#PAR#PlotDen'),
            efficiency = cms.string('RecoTauV/miniAODValidation/ZMM/Summary/#PAR#Plot'),
            numerator = cms.string('RecoTauV/miniAODValidation/ZMM/Summary/#PAR#PlotNum'),
            parameter = cms.vstring('summary'),
            stepByStep = cms.bool(True)
        ),
    )
)
efficienciesTauValidationMiniAODQCD = cms.EDProducer("TauDQMHistEffProducer",
    plots = cms.PSet(
        Summary = cms.PSet(
            denominator = cms.string('RecoTauV/miniAODValidation/QCD/Summary/#PAR#PlotDen'),
            efficiency = cms.string('RecoTauV/miniAODValidation/QCD/Summary/#PAR#Plot'),
            numerator = cms.string('RecoTauV/miniAODValidation/QCD/Summary/#PAR#PlotNum'),
            parameter = cms.vstring('summary'),
            stepByStep = cms.bool(True)
        ),
    )
)
efficienciesTauValidationMiniAODRealData = cms.EDProducer("TauDQMHistEffProducer",
    plots = cms.PSet(
        Summary = cms.PSet(
            denominator = cms.string('RecoTauV/miniAODValidation/JETHT/Summary/#PAR#PlotDen'),
            efficiency = cms.string('RecoTauV/miniAODValidation/JETHT/Summary/#PAR#Plot'),
            numerator = cms.string('RecoTauV/miniAODValidation/JETHT/Summary/#PAR#PlotNum'),
            parameter = cms.vstring('summary'),
            stepByStep = cms.bool(True)
        ),
    )
)
efficienciesTauValidationMiniAODRealElectronsData = cms.EDProducer("TauDQMHistEffProducer",
    plots = cms.PSet(
        Summary = cms.PSet(
            denominator = cms.string('RecoTauV/miniAODValidation/DoubleElectron/Summary/#PAR#PlotDen'),
            efficiency = cms.string('RecoTauV/miniAODValidation/DoubleElectron/Summary/#PAR#Plot'),
            numerator = cms.string('RecoTauV/miniAODValidation/DoubleElectron/Summary/#PAR#PlotNum'),
            parameter = cms.vstring('summary'),
            stepByStep = cms.bool(True)
        ),
    )
)
efficienciesTauValidationMiniAODRealMuonsData = cms.EDProducer("TauDQMHistEffProducer",
    plots = cms.PSet(
        Summary = cms.PSet(
            denominator = cms.string('RecoTauV/miniAODValidation/DoubleMuon/Summary/#PAR#PlotDen'),
            efficiency = cms.string('RecoTauV/miniAODValidation/DoubleMuon/Summary/#PAR#Plot'),
            numerator = cms.string('RecoTauV/miniAODValidation/DoubleMuon/Summary/#PAR#PlotNum'),
            parameter = cms.vstring('summary'),
            stepByStep = cms.bool(True)
        ),
    )
)

tauValidationSequenceMiniAOD = cms.Sequence(
    produceDenominatorZTT
    *tauValidationMiniAODZTT
    *produceDenominatorZEE
    *tauValidationMiniAODZEE
    *produceDenominatorZMM
    *tauValidationMiniAODZMM
    *produceDenominatorQCD
    *tauValidationMiniAODQCD
    *tauValidationMiniAODRealData
    *tauValidationMiniAODRealElectronsData
    *tauValidationMiniAODRealMuonsData
)
from Configuration.Eras.Modifier_fastSim_cff import fastSim
fastSim.toReplaceWith(tauValidationSequenceMiniAOD,tauValidationSequenceMiniAOD.copyAndExclude([tauValidationMiniAODRealData,tauValidationMiniAODRealElectronsData,tauValidationMiniAODRealMuonsData]))

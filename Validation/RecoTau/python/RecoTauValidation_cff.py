import FWCore.ParameterSet.Config as cms

from Validation.RecoTau.TauValidator import TauValidator as _TauValidator

# recoTauValidation = _TauValidator(
#     recoTauCollection = "hpsPFTauProducer",
#     genTauCollection = "tauGenJetsSelectorAllHadrons", # only GenTaus decaying hadronically
#     recoTauIDCollections = [], # reco discriminators not available at RECO level (only at MINI)
#     cutIDs_wp = [], # WP discriminator (disabled if < 0)
#     cutIDs_raw = [], # raw discriminator value cuts (disabled if 0.0)
#     minDeltaR = 0.3,
#     outFolder = "Tau/TauValidation",
#     isPatTaus = False
# )

recoTauValidation = _TauValidator(
    recoTauCollection = "slimmedTausNoDeepIDs",
    genTauCollection = "tauGenJetsSelectorAllHadrons", # only GenTaus decaying hadronically
    recoTauIDCollections = ["deepTau2026v2p5ForMini:VSjet", "deepTau2026v2p5ForMini:VSe", "deepTau2026v2p5ForMini:VSmu"],
    cutIDs_wp = [-1, -1, -1], # WP discriminator (disabled if < 0)
    cutIDs_raw = [0.0, 0.0, 0.0], # raw discriminator value cuts (disabled if 0.0)
    minDeltaR = 0.3,
    outFolder = "Tau/TauValidation",
    isPatTaus = True
)

recoTauValidation_cutWPVsJet_0 = recoTauValidation.clone(cutIDs_wp = [0, -1, -1])
recoTauValidation_cutWPVsJet_1 = recoTauValidation.clone(cutIDs_wp = [1, -1, -1])
recoTauValidation_cutWPVsJet_2 = recoTauValidation.clone(cutIDs_wp = [2, -1, -1])
recoTauValidation_cutWPVsJet_3 = recoTauValidation.clone(cutIDs_wp = [3, -1, -1])

recoTauValidation_cutIdVsJet_0p5 = recoTauValidation.clone(cutIDs_raw = [0.5, -1, -1])
recoTauValidation_cutIdVsJet_0p7 = recoTauValidation.clone(cutIDs_raw = [0.7, -1, -1])
recoTauValidation_cutIdVsJet_0p9 = recoTauValidation.clone(cutIDs_raw = [0.9, -1, -1])
recoTauValidation_cutIdVsJet_0p95 = recoTauValidation.clone(cutIDs_raw = [0.95, -1, -1])
recoTauValidation_cutIdVsJet_0p99 = recoTauValidation.clone(cutIDs_raw = [0.99, -1, -1])

recoTauValidation_deltaR0p3 = recoTauValidation.clone(
    minDeltaR = 0.3,
    outFolder = "Tau/TauValidation_DeltaR/DeltaR0p3",
)

recoTauValidation_deltaR0p25 = recoTauValidation.clone(
    minDeltaR = 0.25,
    outFolder = "Tau/TauValidation_DeltaR/DeltaR0p25",
)

recoTauValidation_deltaR0p2 = recoTauValidation.clone(
    minDeltaR = 0.2,
    outFolder = "Tau/TauValidation_DeltaR/DeltaR0p2",
)

recoTauValidation_deltaR0p15 = recoTauValidation.clone(
    minDeltaR = 0.15,
    outFolder = "Tau/TauValidation_DeltaR/DeltaR0p15",
)

recoTauValidation_deltaR0p1 = recoTauValidation.clone(
    minDeltaR = 0.1,
    outFolder = "Tau/TauValidation_DeltaR/DeltaR0p1",
)

recoTauValidation_deltaR = cms.Sequence(
    recoTauValidation_deltaR0p3 +
    recoTauValidation_deltaR0p25 +
    recoTauValidation_deltaR0p2 +
    recoTauValidation_deltaR0p15 +
    recoTauValidation_deltaR0p1
)

recoTauValidation_wp = cms.Sequence(
    recoTauValidation_cutWPVsJet_0
    + recoTauValidation_cutWPVsJet_1
    + recoTauValidation_cutWPVsJet_2
    + recoTauValidation_cutWPVsJet_3
)

recoTauValidation_id = cms.Sequence(
    recoTauValidation_cutIdVsJet_0p5
    + recoTauValidation_cutIdVsJet_0p7
    + recoTauValidation_cutIdVsJet_0p9
    + recoTauValidation_cutIdVsJet_0p95
    + recoTauValidation_cutIdVsJet_0p99
)


recoTauValidationSequence = cms.Sequence(
    recoTauValidation
    # WP scanning
    + recoTauValidation_wp
    # ID scanning
    + recoTauValidation_id
    # DeltaR scanning
    # + recoTauValidation_deltaR
)

# Old Run-3 validation, not maintained for Phase-2

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

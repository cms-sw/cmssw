from Validation.RecoTau.dataTypes.ValidateTausOnRealData_cff import *
from Validation.RecoTau.dataTypes.ValidateTausOnRealElectronsData_cff import *
from Validation.RecoTau.dataTypes.ValidateTausOnRealMuonsData_cff import *

pfTauRunDQMValidation = cms.Sequence(
    TauValNumeratorAndDenominatorRealData+
    TauValNumeratorAndDenominatorRealElectronsData+
    TauValNumeratorAndDenominatorRealMuonsData
    )

produceDenoms = cms.Sequence(
    produceDenominatorRealData+
    produceDenominatorRealElectronsData+
    produceDenominatorRealMuonsData
    )

runTauEff = cms.Sequence(
    efficienciesRealData+
    efficienciesRealElectronsData+
    efficienciesRealMuonsData+
    normalizePlotsRealMuonsData
    )

runTauEffSingleMu = cms.Sequence(
        efficienciesRealMuonsData+
            efficienciesRealData+
            normalizePlotsRealMuonsData
            )

runTauEffJet = cms.Sequence(
        TauEfficienciesRealData
            )

runTauEffMutiJet = runTauEffJet

runTauEffSingleE = cms.Sequence(
        produceDenominatorRealElectronsData
            )
runTauEffTausPlusE = runTauEffSingleE

##Full sequences, including normalizations
## TauEfficienciesRealData+
## TauEfficienciesRealElectronsData+
## TauEfficienciesRealMuonsData


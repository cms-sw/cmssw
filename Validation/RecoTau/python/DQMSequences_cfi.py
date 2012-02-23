from Validation.RecoTau.dataTypes.ValidateTausOnRealData_cff import *
from Validation.RecoTau.dataTypes.ValidateTausOnRealElectronsData_cff import *
from Validation.RecoTau.dataTypes.ValidateTausOnRealMuonsData_cff import *

dqmInfoTauV = cms.EDAnalyzer(
    "DQMEventInfo",
    subSystemFolder = cms.untracked.string('RecoTauV')
    )


produceDenoms = cms.Sequence(
    produceDenominatorRealData+
    produceDenominatorRealElectronsData+
    produceDenominatorRealMuonsData
    )

pfTauRunDQMValidation = cms.Sequence(
    TauValNumeratorAndDenominatorRealData+
    TauValNumeratorAndDenominatorRealElectronsData+
    TauValNumeratorAndDenominatorRealMuonsData+
    dqmInfoTauV
    )

runTauEff = cms.Sequence(
    efficienciesRealData+
    efficienciesRealElectronsData+
    efficienciesRealMuonsData+
    normalizePlotsRealMuonsData
    )

#Denominators according to dataset
produceDenomsMu = cms.Sequence(
    produceDenominatorRealData+
    produceDenominatorRealMuonsData
    )
produceDenomsSingleMu = produceDenomsMu

produceDenomsJet = produceDenominatorRealData
produceDenomsMultiJet = produceDenomsJet

produceDenomsDoubleElectron = produceDenominatorRealElectronsData
produceDenomsTauPlusX = produceDenomsDoubleElectron

#Main modules according to dataset
pfTauRunDQMValidationMu = cms.Sequence(
    TauValNumeratorAndDenominatorRealData+
    TauValNumeratorAndDenominatorRealMuonsData+
    dqmInfoTauV
    )
pfTauRunDQMValidationSingleMu = pfTauRunDQMValidationMu

pfTauRunDQMValidationJet = cms.Sequence(
    TauValNumeratorAndDenominatorRealData+
    dqmInfoTauV
    )
pfTauRunDQMValidationMultiJet = pfTauRunDQMValidationJet

pfTauRunDQMValidationDoubleElectron = cms.Sequence(
    TauValNumeratorAndDenominatorRealElectronsData+
    dqmInfoTauV
    )
pfTauRunDQMValidationTauPlusX = pfTauRunDQMValidationDoubleElectron

#Efficiencies production according to dataset
runTauEffMu = cms.Sequence(
    efficienciesRealMuonsData+
    efficienciesRealData+
    normalizePlotsRealMuonsData
    )
runTauEffSingleMu = runTauEffMu

runTauEffJet = TauEfficienciesRealData
runTauEffMutiJet = runTauEffJet

runTauEffDoubleElectron = produceDenominatorRealElectronsData
runTauEffTauPlusX = runTauEffDoubleElectron

##Full sequences, including normalizations
## TauEfficienciesRealData+
## TauEfficienciesRealElectronsData+
## TauEfficienciesRealMuonsData


from Validation.RecoTau.dataTypes.ValidateTausOnRealData_cff import *
from Validation.RecoTau.dataTypes.ValidateTausOnRealElectronsData_cff import *
from Validation.RecoTau.dataTypes.ValidateTausOnRealMuonsData_cff import *

produceDenoms = cms.Sequence(
    produceDenominatorRealData+
    produceDenominatorRealElectronsData+
    produceDenominatorRealMuonsData
    )

pfTauRunDQMValidation = cms.Sequence(
    TauValNumeratorAndDenominatorRealData+
    TauValNumeratorAndDenominatorRealElectronsData+
    TauValNumeratorAndDenominatorRealMuonsData
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
    TauValNumeratorAndDenominatorRealMuonsData
    )
pfTauRunDQMValidationSingleMu = pfTauRunDQMValidationMu

pfTauRunDQMValidationJet = TauValNumeratorAndDenominatorRealData
pfTauRunDQMValidationMultiJet = pfTauRunDQMValidationJet

pfTauRunDQMValidationDoubleElectron = TauValNumeratorAndDenominatorRealElectronsData
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


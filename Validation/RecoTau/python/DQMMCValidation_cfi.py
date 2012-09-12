from Validation.RecoTau.dataTypes.ValidateTausOnQCD_cff import *
from Validation.RecoTau.dataTypes.ValidateTausOnRealData_cff import *
from Validation.RecoTau.dataTypes.ValidateTausOnRealElectronsData_cff import *
from Validation.RecoTau.dataTypes.ValidateTausOnRealMuonsData_cff import *
from Validation.RecoTau.dataTypes.ValidateTausOnZEEFastSim_cff import *
from Validation.RecoTau.dataTypes.ValidateTausOnZEE_cff import *
from Validation.RecoTau.dataTypes.ValidateTausOnZMM_cff import *
from Validation.RecoTau.dataTypes.ValidateTausOnZTTFastSim_cff import *
from Validation.RecoTau.dataTypes.ValidateTausOnZTT_cff import *

pfTauRunDQMValidation = cms.Sequence(
    TauValNumeratorAndDenominatorQCD+
    TauValNumeratorAndDenominatorRealData+
    TauValNumeratorAndDenominatorRealElectronsData+
    TauValNumeratorAndDenominatorRealMuonsData+
    TauValNumeratorAndDenominatorZEE+
    TauValNumeratorAndDenominatorZMM+
    TauValNumeratorAndDenominatorZTT
    )

produceDenoms = cms.Sequence(
    produceDenominatorQCD+
    produceDenominatorRealData+
    produceDenominatorRealElectronsData+
    produceDenominatorRealMuonsData+
    produceDenominatorZEE+
    produceDenominatorZMM+
    produceDenominatorZTT
    )

runTauEff = cms.Sequence(
    efficienciesQCD+
    efficienciesRealData+
    efficienciesRealElectronsData+
    efficienciesRealMuonsData+
    efficienciesZEE+
    efficienciesZMM+
    efficienciesZTT+
    normalizePlotsZTT
    )
##Full sequences, including normalizations
## TauEfficienciesQCD+
## TauEfficienciesRealData+
## TauEfficienciesRealElectronsData+
## TauEfficienciesRealMuonsData+
## TauEfficienciesZEEFastSim+
## TauEfficienciesZEE+
## TauEfficienciesZMM+
## TauEfficienciesZTTFastSim+
## TauEfficienciesZTT


makeBetterPlots = cms.Sequence() #Not used anymore/by now

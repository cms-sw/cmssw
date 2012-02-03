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
    TauValNumeratorAndDenominatorZEEFastSim+
    TauValNumeratorAndDenominatorZEE+
    TauValNumeratorAndDenominatorZMM+
    TauValNumeratorAndDenominatorZTTFastSim+
    TauValNumeratorAndDenominatorZTT
    )

produceDenoms = cms.Sequence(
    produceDenominatorQCD+
    produceDenominatorRealData+
    produceDenominatorRealElectronsData+
    produceDenominatorRealMuonsData+
    produceDenominatorZEEFastSim+
    produceDenominatorZEE+
    produceDenominatorZMM+
    produceDenominatorZTTFastSim+
    produceDenominatorZTT
    )

runTauEff = cms.Sequence(
    TauEfficienciesQCD+
    TauEfficienciesRealData+
    TauEfficienciesRealElectronsData+
    TauEfficienciesRealMuonsData+
    TauEfficienciesZEEFastSim+
    TauEfficienciesZEE+
    TauEfficienciesZMM+
    TauEfficienciesZTTFastSim+
    TauEfficienciesZTT
    )


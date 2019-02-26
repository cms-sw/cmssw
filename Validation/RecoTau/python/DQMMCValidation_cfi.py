
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
from Configuration.Eras.Modifier_phase1Pixel_cff import phase1Pixel

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
    efficienciesQCDSummary+
    efficienciesRealData+
    efficienciesRealDataSummary+
    efficienciesRealElectronsData+
    efficienciesRealElectronsDataSummary+
    efficienciesRealMuonsData+
    efficienciesRealMuonsDataSummary+
    efficienciesZEE+
    efficienciesZEESummary+
    efficienciesZMM+
    efficienciesZMMSummary+
    efficienciesZTT+
    efficienciesZTTSummary+
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

#dataTypes related to real data are imported from DQMOffline/PFTau
#to avoid duplication, common part of the code between Data & MC are kept
#in DQMOffline/PFTau

from Configuration.StandardSequences.Eras import eras

from Validation.RecoTau.dataTypes.ValidateTausOnQCD_cff import *
from DQMOffline.PFTau.dataTypes.ValidateTausOnRealData_cff import *
from DQMOffline.PFTau.dataTypes.ValidateTausOnRealElectronsData_cff import *
from DQMOffline.PFTau.dataTypes.ValidateTausOnRealMuonsData_cff import *
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
eras.phase1Pixel.toReplaceWith(pfTauRunDQMValidation, pfTauRunDQMValidation.copyAndExclude([ # FIXME
    TauValNumeratorAndDenominatorRealData,          # Excessive printouts because 2017 doesn't have HLT yet
    TauValNumeratorAndDenominatorRealElectronsData, # Excessive printouts because 2017 doesn't have HLT yet
    TauValNumeratorAndDenominatorRealMuonsData,     # Excessive printouts because 2017 doesn't have HLT yet
]))

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

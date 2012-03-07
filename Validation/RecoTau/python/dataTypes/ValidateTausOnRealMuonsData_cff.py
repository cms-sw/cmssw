import FWCore.ParameterSet.Config as cms
from Validation.RecoTau.RecoTauValidation_cfi import *
import copy

from RecoJets.Configuration.RecoPFJets_cff import *
import PhysicsTools.PatAlgos.tools.helpers as helpers

kinematicSelectedTauValDenominatorRealMuonsData = cms.EDFilter( ##FIXME: this should be a filter
   "TauValMuonSelector", #"GenJetSelector"
   src = cms.InputTag("muons"),
   cut = cms.string(kinematicSelectedTauValDenominatorCut.value()+' && isGlobalMuon && isTrackerMuon && isPFIsolationValid && (pfIsolationR04.sumChargedParticlePt + max(pfIsolationR04.sumPhotonEt + pfIsolationR04.sumNeutralHadronEt - 0.0729*pfIsolationR04.sumPUPt,0.0) )/pt < 0.25'),#cms.string('pt > 5. && abs(eta) < 2.5'), #Defined: Validation.RecoTau.RecoTauValidation_cfi 
   filter = cms.bool(False)
)

procAttributes = dir(proc) #Takes a snapshot of what there in the process
helpers.cloneProcessingSnippet( proc, proc.TauValNumeratorAndDenominator, 'RealMuonsData') #clones the sequence inside the process with RealMuonsData postfix
helpers.cloneProcessingSnippet( proc, proc.TauEfficiencies, 'RealMuonsData') #clones the sequence inside the process with RealMuonsData postfix
helpers.massSearchReplaceAnyInputTag(proc.TauValNumeratorAndDenominatorRealMuonsData, 'kinematicSelectedTauValDenominator', 'kinematicSelectedTauValDenominatorRealMuonsData') #sets the correct input tag

#adds to TauValNumeratorAndDenominator modules in the sequence RealMuonsData to the extention name
zttLabeler = lambda module : SetValidationExtention(module, 'RealMuonsData')
zttModifier = ApplyFunctionToSequence(zttLabeler)
proc.TauValNumeratorAndDenominatorRealMuonsData.visit(zttModifier)

#Sets the correct naming to efficiency histograms
proc.efficienciesRealMuonsData.plots = Utils.SetPlotSequence(proc.TauValNumeratorAndDenominatorRealMuonsData)

#checks what's new in the process (the cloned sequences and modules in them)
newProcAttributes = filter( lambda x: (x not in procAttributes) and (x.find('RealMuonsData') != -1), dir(proc) )

#spawns a local variable with the same name as the proc attribute, needed for future process.load
for newAttr in newProcAttributes:
    locals()[newAttr] = getattr(proc,newAttr)

produceDenominatorRealMuonsData = cms.Sequence(
      kinematicSelectedTauValDenominatorRealMuonsData
      )

produceDenominator = produceDenominatorRealMuonsData

runTauValidationBatchMode = cms.Sequence(
      produceDenominator
      +TauValNumeratorAndDenominatorRealMuonsData
      )

runTauValidation = cms.Sequence(
      runTauValidationBatchMode
      +TauEfficienciesRealMuonsData
      )

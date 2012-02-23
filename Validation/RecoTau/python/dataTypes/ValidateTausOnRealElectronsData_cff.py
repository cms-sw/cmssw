import FWCore.ParameterSet.Config as cms
from Validation.RecoTau.RecoTauValidation_cfi import *
import copy

from RecoJets.Configuration.RecoPFJets_cff import *
import PhysicsTools.PatAlgos.tools.helpers as helpers

selectGoodElectrons = cms.EDFilter(
    'ElectronIdFilter',
    src = cms.InputTag('gsfElectrons'),
    eidsrc = cms.InputTag('eidLoose'),
    eid = cms.int32(13)
    )

kinematicSelectedTauValDenominatorRealElectronsData = cms.EDFilter( ##FIXME: this should be a filter
   "TauValElectronSelector", #"GenJetSelector"
   src = cms.InputTag("selectGoodElectrons"),
   cut = cms.string(kinematicSelectedTauValDenominatorCut.value()+' && isElectron && (dr04IsolationVariables.tkSumPt + dr04IsolationVariables.ecalRecHitSumEt )/pt < 0.25'),#cms.string('pt > 5. && abs(eta) < 2.5'), #Defined: Validation.RecoTau.RecoTauValidation_cfi 
   filter = cms.bool(False)
)

procAttributes = dir(proc) #Takes a snapshot of what there in the process
helpers.cloneProcessingSnippet( proc, proc.TauValNumeratorAndDenominator, 'RealElectronsData') #clones the sequence inside the process with RealElectronsData postfix
helpers.cloneProcessingSnippet( proc, proc.TauEfficiencies, 'RealElectronsData') #clones the sequence inside the process with RealElectronsData postfix
helpers.massSearchReplaceAnyInputTag(proc.TauValNumeratorAndDenominatorRealElectronsData, 'kinematicSelectedTauValDenominator', 'kinematicSelectedTauValDenominatorRealElectronsData') #sets the correct input tag

#adds to TauValNumeratorAndDenominator modules in the sequence RealElectronsData to the extention name
zttLabeler = lambda module : SetValidationExtention(module, 'RealElectronsData')
zttModifier = ApplyFunctionToSequence(zttLabeler)
proc.TauValNumeratorAndDenominatorRealElectronsData.visit(zttModifier)

#-----------------------------------------Sets binning
binning = cms.PSet(
    pt = cms.PSet( nbins = cms.int32(8), min = cms.double(0.), max = cms.double(300.) ), #hinfo(75, 0., 150.)
    eta = cms.PSet( nbins = cms.int32(4), min = cms.double(-3.), max = cms.double(3.) ), #hinfo(60, -3.0, 3.0);
    phi = cms.PSet( nbins = cms.int32(4), min = cms.double(-180.), max = cms.double(180.) ), #hinfo(36, -180., 180.);
    pileup = cms.PSet( nbins = cms.int32(16), min = cms.double(0.), max = cms.double(80.) ),#hinfo(25, 0., 25.0);
    )
zttModifier = ApplyFunctionToSequence(lambda m: setBinning(m,binning))
proc.TauValNumeratorAndDenominatorRealElectronsData.visit(zttModifier)
#-----------------------------------------

#Sets the correct naming to efficiency histograms
proc.efficienciesRealElectronsData.plots = Utils.SetPlotSequence(proc.TauValNumeratorAndDenominatorRealElectronsData)

#checks what's new in the process (the cloned sequences and modules in them)
newProcAttributes = filter( lambda x: (x not in procAttributes) and (x.find('RealElectronsData') != -1), dir(proc) )

#spawns a local variable with the same name as the proc attribute, needed for future process.load
for newAttr in newProcAttributes:
    locals()[newAttr] = getattr(proc,newAttr)

produceDenominatorRealElectronsData = cms.Sequence(
    selectGoodElectrons
    *kinematicSelectedTauValDenominatorRealElectronsData
    )

produceDenominator = produceDenominatorRealElectronsData

runTauValidationBatchMode = cms.Sequence(
      produceDenominator
      +TauValNumeratorAndDenominatorRealElectronsData
      )

runTauValidation = cms.Sequence(
      runTauValidationBatchMode
      +TauEfficienciesRealElectronsData
      )

import FWCore.ParameterSet.Config as cms
from Validation.RecoTau.RecoTauValidation_cfi import *
import copy

from RecoJets.Configuration.RecoPFJets_cff import *
import PhysicsTools.PatAlgos.tools.helpers as helpers

selectedMuons = cms.EDFilter("MuonSelector",
    src = cms.InputTag('muons'),
    cut = cms.string("pt > 20.0 && abs(eta) < 2.1 && isGlobalMuon = 1 && isTrackerMuon = 1 && abs(innerTrack().dxy) < 2.0 && abs(innerTrack().dz) < 24."),
    filter = cms.bool(False)
	)

selectedMuonsIso = cms.EDFilter("MuonSelector",
    src = cms.InputTag('selectedMuons'),
    cut = cms.string('(isolationR03().emEt + isolationR03().hadEt + isolationR03().sumPt)/pt < 0.25'),
    filter = cms.bool(False)
	)    

from SimGeneral.HepPDTESSource.pythiapdt_cfi import *

goodTracks = cms.EDFilter("TrackSelector",
    src = cms.InputTag("generalTracks"), 
    cut = cms.string("pt > 5 && abs(eta) < 2.5 && abs(dxy) < 2.0 && abs(dz) < 24."),
    #cut = cms.string("pt > 0.5 "),
    filter = cms.bool(False)
	)

trackCands  = cms.EDProducer("ConcreteChargedCandidateProducer", 
    src  = cms.InputTag("goodTracks"),      
    particleType = cms.string("mu+")     # this is needed to define a mass
	)

ZmmCandMuonTrack = cms.EDProducer("CandViewShallowCloneCombiner",
    decay = cms.string("selectedMuonsIso@+ trackCands@-"), # it takes opposite sign collection, no matter if +- or -+
    cut   = cms.string("60 < mass < 120")
	)

BestZ = cms.EDProducer("BestMassZArbitrationProducer", # returns the Z with mass closest to 91.18 GeV
	ZCandidateCollection = cms.InputTag("ZmmCandMuonTrack")
	)

ZLegs  = cms.EDProducer("CollectionFromZLegProducer", 
    ZCandidateCollection  = cms.InputTag("BestZ"),      
	)

# kinematicSelectedTauValDenominatorRealMuonsData = cms.EDFilter( ##FIXME: this should be a filter
#    "TauValMuonSelector", #"GenJetSelector"
#    src = cms.InputTag("muons"),
#    cut = cms.string(kinematicSelectedTauValDenominatorCut.value()+' && isGlobalMuon && isTrackerMuon && isPFIsolationValid && (pfIsolationR04.sumChargedParticlePt + max(pfIsolationR04.sumPhotonEt + pfIsolationR04.sumNeutralHadronEt - 0.0729*pfIsolationR04.sumPUPt,0.0) )/pt < 0.25'),#cms.string('pt > 5. && abs(eta) < 2.5'), #Defined: Validation.RecoTau.RecoTauValidation_cfi 
#    filter = cms.bool(False)
# )

procAttributes = dir(proc) #Takes a snapshot of what there in the process
helpers.cloneProcessingSnippet( proc, proc.TauValNumeratorAndDenominator, 'RealMuonsData') #clones the sequence inside the process with RealMuonsData postfix
helpers.cloneProcessingSnippet( proc, proc.TauEfficiencies, 'RealMuonsData') #clones the sequence inside the process with RealMuonsData postfix
helpers.massSearchReplaceAnyInputTag(proc.TauValNumeratorAndDenominatorRealMuonsData, 'kinematicSelectedTauValDenominator',  cms.InputTag("ZLegs","theProbeLeg")) #sets the correct input tag

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

produceDenominatorRealMuonsData = cms.Sequence( ( ( selectedMuons * selectedMuonsIso ) +
                                                ( goodTracks * trackCands ) ) *
						                        ZmmCandMuonTrack *
						                        BestZ *
						                        ZLegs 
						                      )

#################################    
# produceDenominatorRealMuonsData = cms.Sequence(
#       kinematicSelectedTauValDenominatorRealMuonsData +
#       #goodMuons *
#       goodTracks +
#       (tagMuons + probeMuons)*
#       ZCandidates
#       #trackCands *
#       )

produceDenominator = produceDenominatorRealMuonsData

runTauValidationBatchMode = cms.Sequence(
      produceDenominator
      +TauValNumeratorAndDenominatorRealMuonsData
      )

runTauValidation = cms.Sequence(
      runTauValidationBatchMode
      +TauEfficienciesRealMuonsData
      )

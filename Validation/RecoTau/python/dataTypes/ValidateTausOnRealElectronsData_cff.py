import FWCore.ParameterSet.Config as cms
from Validation.RecoTau.RecoTauValidation_cfi import *
import copy

from RecoJets.Configuration.RecoPFJets_cff import *
import PhysicsTools.PatAlgos.tools.helpers as helpers

# selectGoodElectrons = cms.EDFilter(
#     'ElectronIdFilter',
#     src = cms.InputTag('gsfElectrons'),
#     eidsrc = cms.InputTag('eidLoose'),
#     eid = cms.int32(13)
#     )
# 
# kinematicSelectedTauValDenominatorRealElectronsData = cms.EDFilter( ##FIXME: this should be a filter
#    "TauValElectronSelector", #"GenJetSelector"
#    src = cms.InputTag("selectGoodElectrons"),
#    cut = cms.string(kinematicSelectedTauValDenominatorCut.value()+' && isElectron && (dr04IsolationVariables.tkSumPt + dr04IsolationVariables.ecalRecHitSumEt )/pt < 0.25'),#cms.string('pt > 5. && abs(eta) < 2.5'), #Defined: Validation.RecoTau.RecoTauValidation_cfi 
#    filter = cms.bool(False)
# )

selectedElectrons = cms.EDFilter(
    "ElectronSelector",
    src = cms.InputTag('gsfElectrons'),
    cut = cms.string("pt > 25.0 && abs(eta) < 2.4 && isElectron && abs(gsfTrack.dxy) < 2. && abs(gsfTrack.dz) < 24."),
    filter = cms.bool(False)
	)

idElectrons = cms.EDFilter(
    "ElectronSelector",
    src = cms.InputTag('selectedElectrons'),
    cut = cms.string('ecalDrivenSeed & isGsfCtfScPixChargeConsistent & isGsfScPixChargeConsistent & isGsfCtfChargeConsistent & !isEBEEGap & (isEB & sigmaIetaIeta<0.01 & abs(deltaPhiSuperClusterTrackAtVtx)<0.06 & abs(deltaEtaSuperClusterTrackAtVtx)<0.006 & hadronicOverEm<0.04 | isEE & sigmaIetaIeta<0.03 & abs(deltaPhiSuperClusterTrackAtVtx)<0.04 & abs(deltaEtaSuperClusterTrackAtVtx)<0.007 & hadronicOverEm<0.025)'),
    filter = cms.bool(False)
)

trackElectrons = cms.EDFilter(
    "ElectronSelector",
    src = cms.InputTag('idElectrons'),
    cut = cms.string('gsfTrack.isNonnull  && 0.7 < eSuperClusterOverP < 1.5'),
#    cut = cms.string('gsfTrack.isNonnull && gsfTrack.trackerExpectedHitsInner.numberOfHits = 0 && 0.7 < eSuperClusterOverP < 1.5'),
    filter = cms.bool(False)
)

isolatedElectrons = cms.EDFilter(
    "ElectronSelector",
    src = cms.InputTag('trackElectrons'),
    cut = cms.string("(isEB & (dr04TkSumPt/pt + max(0.,dr04EcalRecHitSumEt-2.)/pt + dr04HcalTowerSumEt/pt < 0.10)) | (isEE & (dr04TkSumPt/pt + dr04EcalRecHitSumEt/pt + dr04HcalTowerSumEt/pt < 0.09))"),
    filter = cms.bool(False)
	)

from SimGeneral.HepPDTESSource.pythiapdt_cfi import *

goodTracks = cms.EDFilter(
    "TrackSelector",
    src = cms.InputTag("generalTracks"), 
    cut = cms.string("pt > 5 && abs(eta) < 2.5 && abs(dxy) < 2.0 && abs(dz) < 24."),
    filter = cms.bool(False)
	)

trackCands  = cms.EDProducer(
    "ConcreteChargedCandidateProducer", 
    src  = cms.InputTag("goodTracks"),      
    particleType = cms.string("e+")     # this is needed to define a mass. Do not trust the sign, it is dummy
	)

ZeeCandElectronTrack = cms.EDProducer("CandViewShallowCloneCombiner",
    decay = cms.string("isolatedElectrons@+ trackCands@-"), # it takes opposite sign collection, no matter if +- or -+
#    cut   = cms.string("60 < mass < 120 && 1. < deltaR(daughter(0).eta,daughter(0).phi,daughter(1).eta,daughter(1).phi) < 4. && deltaPhi(daughter(0).phi,daughter(1).phi) > 1.")
    cut   = cms.string("60 < mass < 120")
	)

BestZ = cms.EDProducer("BestMassZArbitrationProducer", # returns the Z with mass closest to 91.18 GeV
	ZCandidateCollection = cms.InputTag("ZeeCandElectronTrack")
	)

ZLegs  = cms.EDProducer("CollectionFromZLegProducer", 
    ZCandidateCollection  = cms.InputTag("BestZ"),      
	)

procAttributes = dir(proc) #Takes a snapshot of what there in the process
helpers.cloneProcessingSnippet( proc, proc.TauValNumeratorAndDenominator, 'RealElectronsData') #clones the sequence inside the process with RealElectronsData postfix
helpers.cloneProcessingSnippet( proc, proc.TauEfficiencies, 'RealElectronsData') #clones the sequence inside the process with RealElectronsData postfix
helpers.massSearchReplaceAnyInputTag(proc.TauValNumeratorAndDenominatorRealElectronsData, 'kinematicSelectedTauValDenominator', cms.InputTag("ZLegs","theProbeLeg")) #sets the correct input tag

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

# produceDenominatorRealElectronsData = cms.Sequence(
#     selectGoodElectrons
#     *kinematicSelectedTauValDenominatorRealElectronsData
#     )

produceDenominatorRealElectronsData = cms.Sequence(
      ( (selectedElectrons * idElectrons * trackElectrons * isolatedElectrons) +
      (goodTracks * trackCands) ) *
      ZeeCandElectronTrack *
      BestZ *
      ZLegs
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

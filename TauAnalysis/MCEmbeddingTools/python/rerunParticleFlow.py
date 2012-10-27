# -*- coding: utf-8 -*-

import FWCore.ParameterSet.Config as cms

import PhysicsTools.PatAlgos.tools.helpers as configtools
from Configuration.EventContent.EventContent_cff import RECOEventContent

import re

def updateInputTags(object, inputProcess):
    if isinstance(object, cms.InputTag):
        #print "InputTag: %s" % object
        keepStatement_regex = r"keep [a-zA-Z0-9*]+_(?P<label>[a-zA-Z0-9*]+)_[a-zA-Z0-9*]+_[a-zA-Z0-9*]+"
        keepStatement_matcher = re.compile(keepStatement_regex)
        isStoredInRECO = False
        for keepStatement in RECOEventContent.outputCommands:
            keepStatement_match = keepStatement_matcher.match(keepStatement)
            if keepStatement_match:
                label = keepStatement_match.group('label')
                if label == object.getModuleLabel():
                    isStoredInRECO = True
        if isStoredInRECO:
            if object.getProcessName() != inputProcess:
                #print "InputTag: %s --> updating processName = %s" % (object, inputProcess)
                object.setProcessName(inputProcess)        
    elif isinstance(object, cms.PSet):
        for attrName in dir(object):
            attr = getattr(object, attrName)
            updateInputTags(attr, inputProcess)
    elif isinstance(object, cms.VPSet):
        for pset in object:
            for attrName in dir(pset):
                attr = getattr(pset, attrName)
                updateInputTags(attr, inputProcess)

from FWCore.ParameterSet.Modules import _Module
class SequenceVisitor():
    def __init__(self, inputProcess):
        self.inputProcess = inputProcess

    def giveNext(self):
	return self.nextInChain
    def givePrev(self):
	return self.prevInChain
      
    def enter(self, visitee):
	if isinstance(visitee, cms.Sequence):
            #print "Sequence: %s" %  visitee.label()
            sequenceVisitor = SequenceVisitor(self.inputProcess)
            visitee.visit(sequenceVisitor)
        elif isinstance(visitee, _Module):
            #print "Module: %s" %  visitee.label()
            for attrName in dir(visitee):
                attr = getattr(visitee, attrName)
                updateInputTags(attr, self.inputProcess)
	      	
    def leave(self, visitee):
        pass

def rerunParticleFlow(process, inputProcess):

    #print "<rerunParticleFlow>:"

    # load conditions needed to run particle-flow sequence
    process.load("RecoLocalCalo.EcalRecAlgos.EcalSeverityLevelESProducer_cfi")
    process.load("RecoLocalCalo.HcalRecAlgos.hcalRecAlgoESProd_cfi")

    process.rerunParticleFlowSequence = cms.Sequence()

    # produce objects needed as input of particle-flow algorithm
    process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAny_cfi")
    process.load("CalibTracker.SiPixelESProducers.SiPixelTemplateDBObjectESProducer_cfi")
    process.UpdaterService = cms.Service("UpdaterService")
    # CV: first part of muon reconstruction (input to particle-flow algorithm)
    if not hasattr(process, "muonreco"):
        process.load("RecoMuon.Configuration.RecoMuon_cff")
    process.rerunParticleFlowSequence += process.muonreco
    process.rerunParticleFlowSequence += process.muIsolation
    process.rerunParticleFlowSequence += process.muonSelectionTypeSequence
    process.load("RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi")
    process.rerunParticleFlowSequence += process.cosmicsMuonIdSequence

    if not hasattr(process, "particleFlowCluster"):
        process.load('RecoParticleFlow.PFClusterProducer.particleFlowCluster_cff')
    process.rerunParticleFlowSequence += process.particleFlowCluster
    if not hasattr(process, "pfTrackingGlobalReco"):
        process.load('RecoParticleFlow.PFTracking.particleFlowTrack_cff')
    process.rerunParticleFlowSequence += process.pfTrackingGlobalReco

    if not hasattr(process, "gsfEcalDrivenElectronSequence"):
        process.load("RecoEgamma.EgammaElectronProducers.gsfElectronSequence_cff")
        process.load("RecoEcal.EgammaCoreTools.EcalNextToDeadChannelESProducer_cff")
    process.rerunParticleFlowSequence += process.gsfEcalDrivenElectronSequence

    # reconstruct PFCandidates
    if not hasattr(process, "particleFlowReco"):
        process.load('RecoParticleFlow.Configuration.RecoParticleFlow_cff')
    process.rerunParticleFlowSequence += process.particleFlowReco

    # CV: second part of muon reconstruction (needs particle-flow output in order to compute muon isolation)
    if not hasattr(process, "muonPFIsolationSequence"):
        process.load("RecoMuon.MuonIsolation.muonPFIsolationValues_cff")
    process.rerunParticleFlowSequence += process.muonPFIsolationSequence
    process.rerunParticleFlowSequence += process.muonshighlevelreco

    # CV: compute MET corrections
    if not hasattr(process, "muonMETValueMapProducer"):
        process.load("RecoMET.METProducers.MuonMETValueMapProducer_cff")
    process.rerunParticleFlowSequence += process.muonMETValueMapProducer
    if not hasattr(process, "muonTCMETValueMapProducer"):
        process.load("RecoMET.METProducers.MuonTCMETValueMapProducer_cff")
    process.rerunParticleFlowSequence += process.muonTCMETValueMapProducer
    
    # build final particleFlow collection
    process.rerunParticleFlowSequence += process.particleFlowLinks

    configtools.cloneProcessingSnippet(process, process.rerunParticleFlowSequence, "ForPFMuonCleaning")
    rerunParticleFlowSequenceForPFMuonCleaning = getattr(process, "rerunParticleFlowSequenceForPFMuonCleaning")

    # CV: update input tags to run on inputProcess
    sequenceVisitor = SequenceVisitor(inputProcess)
    process.rerunParticleFlowSequenceForPFMuonCleaning.visit(sequenceVisitor)
    # CV: update instanceLabel dynamically composed from moduleLabel by MuonProducer
    process.particleFlowForPFMuonCleaning.Muons = cms.InputTag("muonsForPFMuonCleaning", "muons1stStepForPFMuonCleaning2muonsForPFMuonCleaninsMap")
            
    return rerunParticleFlowSequenceForPFMuonCleaning

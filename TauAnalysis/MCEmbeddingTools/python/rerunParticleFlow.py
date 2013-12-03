# -*- coding: utf-8 -*-

import FWCore.ParameterSet.Config as cms

import PhysicsTools.PatAlgos.tools.helpers as configtools
from Configuration.EventContent.EventContent_cff import RECOEventContent

import re

def loadIfNecessary(process, configFile, module_or_sequenceName):
    if not hasattr(process, module_or_sequenceName):
        process.load(configFile)

from FWCore.ParameterSet.Modules import _Module
class seqVisitorGetModuleNames():
    def __init__(self):
        self.modulesNames = []

    def giveNext(self):
	return self.nextInChain
    def givePrev(self):
	return self.prevInChain
      
    def enter(self, visitee):
        if isinstance(visitee, _Module):
            self.modulesNames.append(visitee.label())
	      	
    def leave(self, visitee):
        pass

    def getModuleNames(self):
        return self.modulesNames

def updateInputTags(process, object, inputProcess):
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
        isInRerunParticleFlowSequence = False
        if hasattr(process, "rerunParticleFlowSequenceForPFMuonCleaning"):
            sequenceVisitor = seqVisitorGetModuleNames()
            getattr(process, "rerunParticleFlowSequenceForPFMuonCleaning").visit(sequenceVisitor)
            moduleNames = sequenceVisitor.getModuleNames()
            for moduleName in moduleNames:
                if moduleName == object.getModuleLabel():
                    isInRerunParticleFlowSequence = True
        if isStoredInRECO and not isInRerunParticleFlowSequence:
            if object.getProcessName() != inputProcess:
                #print "InputTag: %s --> updating processName = %s" % (object, inputProcess)
                object.setProcessName(inputProcess)        
    elif isinstance(object, cms.PSet):
        for attrName in dir(object):
            attr = getattr(object, attrName)
            updateInputTags(process, attr, inputProcess)
    elif isinstance(object, cms.VPSet):
        for pset in object:
            for attrName in dir(pset):
                attr = getattr(pset, attrName)
                updateInputTags(process, attr, inputProcess)

def rerunParticleFlow(process, inputProcess):

    # load event-setup definitions necessary to rerun particle-flow sequence
    loadIfNecessary(process, "TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAny_cfi", "SteppingHelixPropagatorAny")
    loadIfNecessary(process, "CalibTracker.SiPixelESProducers.SiPixelTemplateDBObjectESProducer_cfi", "siPixelTemplateDBObjectESProducer")
    loadIfNecessary(process, "RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi", "MeasurementTracker")
    loadIfNecessary(process, "RecoLocalCalo.EcalRecAlgos.EcalSeverityLevelESProducer_cfi", "ecalSeverityLevel")
    loadIfNecessary(process, "RecoEcal.EgammaCoreTools.EcalNextToDeadChannelESProducer_cff", "ecalNextToDeadChannelESProducer")
    loadIfNecessary(process, "RecoLocalCalo.HcalRecAlgos.hcalRecAlgoESProd_cfi", "hcalRecAlgos")

    if not hasattr(process, "UpdaterService"):
        process.UpdaterService = cms.Service("UpdaterService")

    # load module definitions necessary to rerun particle-flow sequence
    loadIfNecessary(process, "RecoParticleFlow.PFClusterProducer.particleFlowCluster_cff", "particleFlowCluster")    
    loadIfNecessary(process, "RecoEgamma.EgammaElectronProducers.gsfElectronSequence_cff", "gsfEcalDrivenElectronSequence")
    loadIfNecessary(process, "RecoParticleFlow.Configuration.RecoParticleFlow_cff", "particleFlowReco")
    loadIfNecessary(process, "RecoMuon.MuonIsolation.muonPFIsolationValues_cff", "muonPFIsolationSequence")

    # define complete sequence of all modules necessary to rerun particle-flow algorithm
    process.rerunParticleFlowSequence = cms.Sequence(
        process.particleFlowCluster 
       + process.particleFlowTrackWithDisplacedVertex
       + process.gsfEcalDrivenElectronSequence
       + process.particleFlowReco
       + process.particleFlowLinks
    )

    # CV: clone sequence and give it a different name so that particle-flow algorithm
    #     can be run using "official" module labels on embedded event later
    configtools.cloneProcessingSnippet(process, process.rerunParticleFlowSequence, "ForPFMuonCleaning")

    # CV: run particle-flow algorithm on final RECO muon collection
    #    (rather than running muon reconstruction sequence in steps)    
    process.pfTrackForPFMuonCleaning.MuColl = cms.InputTag('muons')
    process.particleFlowBlockForPFMuonCleaning.RecMuons = cms.InputTag('muons')
    process.particleFlowTmpForPFMuonCleaning.muons = cms.InputTag('muons')
    process.particleFlowForPFMuonCleaning.FillMuonRefs = False

    # CV: make sure that all particle-flow based isolation is computed wrt. 'particleFlowTmp' collection
    #    (PAT may overwrite configuration parameters to refer to 'particleFlow' instead)
    configtools.massSearchReplaceAnyInputTag(process.rerunParticleFlowSequenceForPFMuonCleaning, cms.InputTag('particleFlow'), cms.InputTag('particleFlowTmp'))

    return process.rerunParticleFlowSequenceForPFMuonCleaning

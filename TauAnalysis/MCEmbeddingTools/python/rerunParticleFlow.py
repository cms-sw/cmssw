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
        if hasattr(process, "rerunParticleFlowSequence"):
            sequenceVisitor = seqVisitorGetModuleNames()
            getattr(process, "rerunParticleFlowSequence").visit(sequenceVisitor)
            moduleNames = sequenceVisitor.getModuleNames()
            for moduleName in moduleNames:
                if moduleName == object.getModuleLabel():
                    isInRerunParticleFlowSequence = True
        if isStoredInRECO:
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

from FWCore.ParameterSet.Modules import _Module
class seqVisitorUpdateInputTags():
    def __init__(self, process, inputProcess):
        self.process = process
        self.inputProcess = inputProcess

    def giveNext(self):
	return self.nextInChain
    def givePrev(self):
	return self.prevInChain
      
    def enter(self, visitee):
	if isinstance(visitee, cms.Sequence):
            #print "Sequence: %s" %  visitee.label()
            sequenceVisitor = SequenceVisitor(self.process, self.inputProcess)
            visitee.visit(sequenceVisitor)
        elif isinstance(visitee, _Module):
            #print "Module: %s" %  visitee.label()
            for attrName in dir(visitee):
                attr = getattr(visitee, attrName)
                updateInputTags(self.process, attr, self.inputProcess)
	      	
    def leave(self, visitee):
        pass

def rerunParticleFlow(process, inputProcess):

    # configuration for re-running particle-flow sequence taken from
    #  RecoParticleFlow/Configuration/test/RecoToDisplay_cfg.py
    loadIfNecessary(process, "RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi", "siPixelRecHits")
    loadIfNecessary(process, "RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi", "siStripMatchedRecHits")
    loadIfNecessary(process, "RecoParticleFlow.PFClusterProducer.particleFlowCluster_cff", "particleFlowCluster")
    loadIfNecessary(process, "RecoEcal.Configuration.RecoEcal_cff.py", "ecalClusters")
    ##loadIfNecessary(process, "RecoPixelVertexing.Configuration.RecoPixelVertexing_cff", "recopixelvertexing")
    loadIfNecessary(process, "RecoTracker.Configuration.RecoTracker_cff", "ckftracks")
    loadIfNecessary(process, "RecoMuon.Configuration.RecoMuonPPonly_cff", "muonrecoComplete")
    loadIfNecessary(process, "RecoEgamma.Configuration.RecoEgamma_cff", "egammaGlobalReco")
    loadIfNecessary(process, "RecoParticleFlow.PFTracking.particleFlowTrack_cff", "pfTrackingGlobalReco")
    loadIfNecessary(process, "RecoEgamma.Configuration.RecoEgamma_cff", "egammaHighLevelRecoPrePF")
    loadIfNecessary(process, "RecoParticleFlow.Configuration.RecoParticleFlow_cff", "particleFlowReco")
    loadIfNecessary(process, "RecoEgamma.Configuration.RecoEgamma_cff", "egammaHighLevelRecoPostPF")
    loadIfNecessary(process, "RecoMuon.Configuration.RecoMuon_cff", "muonshighlevelreco")
    loadIfNecessary(process, "RecoParticleFlow.Configuration.RecoParticleFlow_cff", "particleFlowLinks")
    process.printEventContent = cms.EDAnalyzer("EventContentAnalyzer")
    process.rerunParticleFlowSequence = cms.Sequence(
        process.siPixelRecHits
       + process.siStripMatchedRecHits
       + process.particleFlowCluster
       + process.ecalClusters
       ##+ process.recopixelvertexing
       + process.ckftracks
       + process.muonrecoComplete
       + process.egammaGlobalReco
       + process.pfTrackingGlobalReco
       + process.egammaHighLevelRecoPrePF
       + process.particleFlowReco
       + process.egammaHighLevelRecoPostPF
       + process.muonshighlevelreco
       + process.printEventContent
       + process.particleFlowLinks
    )

    configtools.cloneProcessingSnippet(process, process.rerunParticleFlowSequence, "ForPFMuonCleaning")
    rerunParticleFlowSequenceForPFMuonCleaning = getattr(process, "rerunParticleFlowSequenceForPFMuonCleaning")

    # CV: update input tags to run on inputProcess
    sequenceVisitor = seqVisitorUpdateInputTags(process, inputProcess)
    process.rerunParticleFlowSequenceForPFMuonCleaning.visit(sequenceVisitor)

    # CV: update moduleLabel passed as string parameter to Multi5x5SuperClusterProducer
    process.multi5x5SuperClustersCleanedForPFMuonCleaning.barrelClusterProducer = cms.string('multi5x5BasicClustersForPFMuonCleaning')
    process.multi5x5SuperClustersCleanedForPFMuonCleaning.endcapClusterProducer = cms.string('multi5x5BasicClustersCleanedForPFMuonCleaning')
    process.multi5x5SuperClustersUncleanedForPFMuonCleaning.barrelClusterProducer = cms.string('multi5x5BasicClustersForPFMuonCleaning')
    process.multi5x5SuperClustersUncleanedForPFMuonCleaning.endcapClusterProducer = cms.string('multi5x5BasicClustersCleanedForPFMuonCleaning')
    # CV: update instanceLabel dynamically composed from moduleLabel by MuonProducer
    process.particleFlowForPFMuonCleaning.Muons = cms.InputTag("muonsForPFMuonCleaning", "muons1stStepForPFMuonCleaning2muonsForPFMuonCleaninsMap")
    # CV: special handling of modules the configuration of which has been overwritten
    #     in order to mix collections of objects reconstructed and Z -> mu+ mu- event with simulated tau decay products
##     if hasattr(process, "trackerDrivenElectronSeeds"):
##         moduleType = getattr(process, "trackerDrivenElectronSeeds").type_()
##         if moduleType == "ElectronSeedTrackRefUpdater":
##             if hasattr(process, "trackerDrivenElectronSeedsORG"):
##                 trackerDrivenElectronSeedsForPFMuonCleaning = getattr(process, "trackerDrivenElectronSeedsORG").clone(
##                     PFPSClusterLabel = cms.InputTag("particleFlowClusterPSForPFMuonCleaning"),
##                     PFHcalClusterLabel = cms.InputTag("particleFlowClusterHCALForPFMuonCleaning"),
##                     PFEcalClusterLabel = cms.InputTag("particleFlowClusterECALForPFMuonCleaning"),
##                     TkColList = cms.VInputTag(cms.InputTag("generalTracks"))
##                 )                
##             else:
##                 raise ValueError ("Failed to find module of type = '%s' replacing 'trackerDrivenElectronSeeds' module !!")
##         elif moduleType == "GoodSeedProducer":
##             pass
##         else:
##             raise ValueError ("Module of name = 'trackerDrivenElectronSeeds' is of invalid type = %s !!" % moduleType)
##     else:
##         raise ValueError ("No module of name = 'trackerDrivenElectronSeeds' defined !!")

    process.convLayerPairs.FPix.HitProducer = cms.string('siPixelRecHitsBLAH')
    process.convLayerPairs.BPix.HitProducer = cms.string('siPixelRecHitsBLAH')
            
    return rerunParticleFlowSequenceForPFMuonCleaning

#!/usr/bin/env python
import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.usedOutput import *


process = cms.Process("SIZE");
process.load("Configuration.StandardSequences.RawToDigi_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.EventContent.EventContent_cff")



#define pretty grouping (could be moved in a cff in the release)
process.PFlow = cms.Sequence(process.particleFlowCluster+process.particleFlowReco)
process.JetMet = cms.Sequence(process.recoJets+process.recoPFMET+process.recoPFJets+process.metrecoPlusHCALNoise+process.recoJetAssociations)
process.TkLocal = cms.Sequence(process.trackerlocalreco)
process.MuLocal = cms.Sequence(process.muonlocalreco)
process.CaloLocal = cms.Sequence(process.calolocalreco+process.caloTowersRec)
process.Ecal = cms.Sequence(process.ecalClusters+process.reducedRecHitsSequence)
process.Tracking = cms.Sequence(process.ckftracks)
process.BTagVtx = cms.Sequence(process.recopixelvertexing+process.vertexreco+process.btagging+process.offlineBeamSpot)
process.Muon = cms.Sequence(process.muonrecoComplete)
process.EGamma = cms.Sequence(process.egammarecoFull+process.electronGsfTracking)
process.Tau = cms.Sequence(process.tautagging+process.PFTau)
names = ["RawToDigi","TkLocal","MuLocal","CaloLocal","Ecal","Tracking","JetMet","BTagVtx","Muon","EGamma","Tau","PFlow"]

def assignModulesToSeqs():
	#assign modules to pretty grouping
	sequenceWithModules = { }
	sequenceWithModulesString = { }
	for name in names:
	    sequenceWithModules[name] = []
	    getModulesFromSequence(process.sequences[name],sequenceWithModules[name])
	    #also create the flat string based version instead of full config one
	    sequenceWithModulesString[name] = []
	    for module in sequenceWithModules[name]:
        	sequenceWithModulesString[name].append(module.label())
	return (sequenceWithModules, sequenceWithModulesString)

if __name__ == "__main__":
	#print
	(sequenceWithModules, sequenceWithModulesString) = assignModulesToSeqs()
	for seq, mods  in sequenceWithModules.items():
	#for seq, mods  in sequenceWithModulesString.items():
	    print "sequence: %s" % (seq)
	    for module in mods:
	        print "  module: %s" % (module)






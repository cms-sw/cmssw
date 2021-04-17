#!/usr/bin/env python


### Various set of customise functions needed for embedding
from __future__ import print_function
import FWCore.ParameterSet.Config as cms

import six

################################ Customizer for skimming ###########################
### There are four different parts.
##First step is the SELECT (former SKIM) part, where we identfy the events which are good for Embedding. Need to store RAWRECO [RAW is needed for the MERG and RECO for the CLEAN step]
##Second step is the CLEAN input are PAT muons with RECO information. After this step only RAW plus some special collections needed for the MERG step must be saved
##Third step is the SIM. The input is the externalLHEProducer, which must be produced before (At the moment we do it parallel to the CLEAN step). Again only save RAW of the SELECT and only save what is need for the MERG step
##Last step is the MERG step. Which is the usally reconstruction, where the input produces are replaced by merg producer, which mix the SIM and CLEAN inputs.

## Some comments on this approach. All steps runs the RECO sequence until the end in the moment. It would be possible to stop after the all inputs which are needed for the MERG step are generated (which happens at a very early state of the reconstruction. But with this approach we are able to get the RECO (and PAT aka miniAOD) of all four step SELECT (orginal event), SIM, CLEAN and MERGED. Therefor one only needs to SAVE the corresponding output (in cmsDriver change output to RAW -> RAW,RECO,PAT)

#######################  Some basic functions ####################
## Helper Class, which summerizes in which step which Producer (Cleaner Merger), should be loaded. It is also usefull to define which collection should be stored for the next step
## E.g What is needed for MERGE must be produce in the CLEAN and SIM step


class module_manipulate():
    def __init__(self, module_name, manipulator_name, steps = ["SELECT","CLEAN","SIM","MERGE"], instance=[""], merge_prefix = ""):
        self.module_name = module_name
        self.manipulator_name = manipulator_name
        self.steps = steps
        self.instance = instance
        self.merger_name = manipulator_name+"ColMerger"
        self.cleaner_name = manipulator_name+"ColCleaner"
        self.merge_prefix = merge_prefix




to_bemanipulate = []


to_bemanipulate.append(module_manipulate(module_name = 'siPixelClusters', manipulator_name = "Pixel", steps = ["SELECT","CLEAN"] ))
to_bemanipulate.append(module_manipulate(module_name = 'siStripClusters', manipulator_name = "Strip", steps = ["SELECT","CLEAN"] ))

to_bemanipulate.append(module_manipulate(module_name = 'generalTracks', manipulator_name = "Track", steps = ["SIM", "MERGE"]))
to_bemanipulate.append(module_manipulate(module_name = 'muons1stStep', manipulator_name = "Muon", steps = ["SIM", "MERGE"]))
to_bemanipulate.append(module_manipulate(module_name = 'gedGsfElectronsTmp', manipulator_name = "GsfElectron", steps = ["SIM", "MERGE"]))
to_bemanipulate.append(module_manipulate(module_name = 'gedPhotonsTmp', manipulator_name = "Photon", steps = ["SIM", "MERGE"]))
to_bemanipulate.append(module_manipulate(module_name = 'particleFlowTmp', manipulator_name = "PF", steps = ["SIM", "MERGE"], instance=["","CleanedHF","CleanedCosmicsMuons","CleanedTrackerAndGlobalMuons","CleanedFakeMuons","CleanedPunchThroughMuons","CleanedPunchThroughNeutralHadrons","AddedMuonsAndHadrons"]))


to_bemanipulate.append(module_manipulate(module_name = 'ecalRecHit', manipulator_name = "EcalRecHit", instance= ["EcalRecHitsEB","EcalRecHitsEE"]))
to_bemanipulate.append(module_manipulate(module_name = 'ecalPreshowerRecHit', manipulator_name = "EcalRecHit", instance= ["EcalRecHitsES"]))

to_bemanipulate.append(module_manipulate(module_name = 'hbheprereco', manipulator_name = "HBHERecHit"))
to_bemanipulate.append(module_manipulate(module_name = 'hbhereco', manipulator_name = "HBHERecHit"))
to_bemanipulate.append(module_manipulate(module_name = 'zdcreco', manipulator_name = "ZDCRecHit"))

to_bemanipulate.append(module_manipulate(module_name = 'horeco', manipulator_name = "HORecHit"))
to_bemanipulate.append(module_manipulate(module_name = 'hfreco', manipulator_name = "HFRecHit"))
to_bemanipulate.append(module_manipulate(module_name = 'castorreco', manipulator_name = "CastorRecHit"))


to_bemanipulate.append(module_manipulate(module_name = 'dt1DRecHits', manipulator_name = "DTRecHit",  steps = ["SELECT","CLEAN"] ))
to_bemanipulate.append(module_manipulate(module_name = 'dt1DCosmicRecHits', manipulator_name = "DTRecHit", steps = ["SELECT","CLEAN"]  ))

to_bemanipulate.append(module_manipulate(module_name = 'csc2DRecHits', manipulator_name = "CSCRecHit", steps = ["SELECT","CLEAN"]  ))
to_bemanipulate.append(module_manipulate(module_name = 'rpcRecHits', manipulator_name = "RPCRecHit",  steps = ["SELECT","CLEAN"] ))


def modify_outputModules(process, keep_drop_list = [], module_veto_list = [] ):
    outputModulesList = [key for key,value in six.iteritems(process.outputModules)]
    for outputModule in outputModulesList:
        if outputModule in module_veto_list:
            continue
        outputModule = getattr(process, outputModule)
        for add_element in keep_drop_list:
            outputModule.outputCommands.extend(add_element)
    return process



################################ Customizer for Selecting ###########################

def keepSelected(dataTier):
    ret_vstring = cms.untracked.vstring(
                  #  "drop *_*_*_"+dataTier,
                    "keep *_patMuonsAfterID_*_"+dataTier,
                    "keep *_slimmedMuons_*_"+dataTier,
                    "keep *_selectedMuonsForEmbedding_*_"+dataTier,
                    "keep recoVertexs_offlineSlimmedPrimaryVertices_*_"+dataTier,
                    "keep *_firstStepPrimaryVertices_*_"+dataTier,
                    "keep *_offlineBeamSpot_*_"+dataTier
                    )
    for akt_manimod in to_bemanipulate:
        if "CLEAN" in akt_manimod.steps:
            ret_vstring.append("keep *_"+akt_manimod.module_name+"_*_"+dataTier)
    return ret_vstring

def customiseSelecting(process,reselect=False):
    if reselect:
        process._Process__name = "RESELECT"
        dataTier="RESELECT"
    else:
        process._Process__name = "SELECT"
        dataTier="SELECT"

    process.load('TauAnalysis.MCEmbeddingTools.SelectingProcedure_cff')
    # don't rekey TrackExtra refs because the full original collections are anyways stored
    process.slimmedMuons.trackExtraAssocs = []
    process.patMuonsAfterKinCuts.src = cms.InputTag("slimmedMuons","",dataTier)
    process.patMuonsAfterID = process.patMuonsAfterLooseID.clone()

    process.selecting = cms.Path(process.makePatMuonsZmumuSelection)
    process.schedule.insert(-1, process.selecting)

    outputModulesList = [key for key,value in six.iteritems(process.outputModules)]
    for outputModule in outputModulesList:
        outputModule = getattr(process, outputModule)
        outputModule.SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring("selecting"))
        outputModule.outputCommands.extend(keepSelected(dataTier))

    process = customisoptions(process)
    return modify_outputModules(process,[keepSelected(dataTier)])

def customiseSelecting_Reselect(process):
    return customiseSelecting(process,reselect=True)

################################ Customizer for cleaining ###########################
def keepCleaned():
    ret_vstring = cms.untracked.vstring(
#	 	                 "drop *_*_*_LHEembeddingCLEAN",
#	 	                 "drop *_*_*_CLEAN"
                            )

    for akt_manimod in to_bemanipulate:
        if "MERGE" in akt_manimod.steps:
            ret_vstring.append("keep *_"+akt_manimod.module_name+"_*_LHEembeddingCLEAN")
            ret_vstring.append("keep *_"+akt_manimod.module_name+"_*_CLEAN")
    ret_vstring.append("keep *_standAloneMuons_*_LHEembeddingCLEAN")
    ret_vstring.append("keep *_glbTrackQual_*_LHEembeddingCLEAN")
    return ret_vstring



def customiseCleaning(process, changeProcessname=True,reselect=False):
    if changeProcessname:
        process._Process__name = "CLEAN"
    if reselect:
        dataTier="RESELECT"
    else: 
        dataTier="SELECT"
    ## Needed for the Calo Cleaner, could also be put into a function wich fix the input parameters
    from TrackingTools.TrackAssociator.default_cfi import TrackAssociatorParameterBlock
    TrackAssociatorParameterBlock.TrackAssociatorParameters.CSCSegmentCollectionLabel = cms.InputTag("cscSegments","",dataTier)
    TrackAssociatorParameterBlock.TrackAssociatorParameters.CaloTowerCollectionLabel = cms.InputTag("towerMaker","",dataTier)
    TrackAssociatorParameterBlock.TrackAssociatorParameters.DTRecSegment4DCollectionLabel = cms.InputTag("dt4DSegments","",dataTier)
    TrackAssociatorParameterBlock.TrackAssociatorParameters.EBRecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEB",dataTier)
    TrackAssociatorParameterBlock.TrackAssociatorParameters.EERecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEE",dataTier)
    TrackAssociatorParameterBlock.TrackAssociatorParameters.HBHERecHitCollectionLabel = cms.InputTag("hbhereco","",dataTier)
    TrackAssociatorParameterBlock.TrackAssociatorParameters.HORecHitCollectionLabel = cms.InputTag("horeco","",dataTier)


    MuonImput = cms.InputTag("selectedMuonsForEmbedding","","")  ## This are the muon
    for akt_manimod in to_bemanipulate:
        if "CLEAN" in akt_manimod.steps:
            oldCollections_in = cms.VInputTag()
            for instance in akt_manimod.instance:
                oldCollections_in.append(cms.InputTag(akt_manimod.module_name,instance,dataTier))
            setattr(process, akt_manimod.module_name, cms.EDProducer(akt_manimod.cleaner_name,MuonCollection = MuonImput,TrackAssociatorParameters = TrackAssociatorParameterBlock.TrackAssociatorParameters,oldCollection = oldCollections_in))
    process.ecalPreshowerRecHit.TrackAssociatorParameters.usePreshower = cms.bool(True)
    process = customisoptions(process)	
    return modify_outputModules(process,[keepSelected(dataTier),keepCleaned()],["MINIAODoutput"])


################################ Customizer for simulaton ###########################
def keepLHE():
    ret_vstring = cms.untracked.vstring()
    ret_vstring.append("keep *_externalLHEProducer_*_LHEembedding")
    ret_vstring.append("keep *_externalLHEProducer_*_LHEembeddingCLEAN")
    return ret_vstring


def keepSimulated():
    ret_vstring = cms.untracked.vstring()
    for akt_manimod in to_bemanipulate:
        if "MERGE" in akt_manimod.steps:
            ret_vstring.append("keep *_"+akt_manimod.module_name+"_*_SIMembedding")
    ret_vstring.append("keep *_genParticles_*_SIMembedding")
    ret_vstring.append("keep *_standAloneMuons_*_SIMembedding")
    ret_vstring.append("keep *_glbTrackQual_*_SIMembedding")
    ret_vstring.append("keep *_generator_*_SIMembedding")
    ret_vstring.append("keep *_addPileupInfo_*_SIMembedding")
    ret_vstring.append("keep *_slimmedAddPileupInfo_*_*")
    return ret_vstring




def customiseLHE(process, changeProcessname=True,reselect=False):
    if reselect:
        dataTier="RESELECT"
    else: 
        dataTier="SELECT"
    if changeProcessname:
        process._Process__name = "LHEembedding"
    process.load('TauAnalysis.MCEmbeddingTools.EmbeddingLHEProducer_cfi')
    if reselect:
        process.externalLHEProducer.vertices=cms.InputTag("offlineSlimmedPrimaryVertices","","RESELECT")
    process.lheproduction = cms.Path(process.makeexternalLHEProducer)
    process.schedule.insert(0,process.lheproduction)


    process = customisoptions(process)
    return modify_outputModules(process,[keepSelected(dataTier),keepCleaned(), keepLHE()],["MINIAODoutput"])


def customiseGenerator(process, changeProcessname=True,reselect=False):
    if reselect:
        dataTier="RESELECT"
    else:
        dataTier="SELECT"
    if changeProcessname:
        process._Process__name = "SIMembedding"

    ## here correct the vertex collection

    process.load('TauAnalysis.MCEmbeddingTools.EmbeddingVertexCorrector_cfi')
    process.VtxSmeared = process.VtxCorrectedToInput.clone()
    print("Correcting Vertex in genEvent to one from input. Replaced 'VtxSmeared' with the Corrector.")

    # Remove BeamSpot Production, use the one from selected data instead.
    process.reconstruction.remove(process.offlineBeamSpot)

    # Disable noise simulation
    process.mix.digitizers.castor.doNoise = cms.bool(False)

    process.mix.digitizers.ecal.doESNoise = cms.bool(False)
    process.mix.digitizers.ecal.doENoise = cms.bool(False)

    process.mix.digitizers.hcal.doNoise = cms.bool(False)
    process.mix.digitizers.hcal.doThermalNoise = cms.bool(False)
    process.mix.digitizers.hcal.doHPDNoise = cms.bool(False)

    process.mix.digitizers.pixel.AddNoisyPixels = cms.bool(False)
    process.mix.digitizers.pixel.AddNoise = cms.bool(False)

    process.mix.digitizers.strip.Noise = cms.bool(False)


    process = customisoptions(process) 
    ##process = fix_input_tags(process)

    return modify_outputModules(process,[keepSelected(dataTier),keepCleaned(),keepSimulated()],["AODSIMoutput"])

def customiseGenerator_Reselect(process):
    return customiseGenerator(process,reselect=True)

################################ Customizer for merging ###########################
def keepMerged(dataTier="SELECT"):
    ret_vstring = cms.untracked.vstring()
    ret_vstring.append("drop *_*_*_"+dataTier)
    ret_vstring.append("keep *_prunedGenParticles_*_MERGE")
    ret_vstring.append("keep *_generator_*_SIMembedding")
    return ret_vstring


def customiseKeepPrunedGenParticles(process,reselect=False):
    if reselect:
        dataTier="RESELECT"
    else:
        dataTier="SELECT"

    process.load('PhysicsTools.PatAlgos.slimming.genParticles_cff')
    process.merge_step += process.prunedGenParticlesWithStatusOne
    process.load('PhysicsTools.PatAlgos.slimming.prunedGenParticles_cfi')
    process.merge_step += process.prunedGenParticles
    process.load('PhysicsTools.PatAlgos.slimming.packedGenParticles_cfi')
    process.merge_step += process.packedGenParticles

    process.load('PhysicsTools.PatAlgos.mcMatchLayer0.muonMatch_cfi')
    process.merge_step += process.muonMatch
    process.load('PhysicsTools.PatAlgos.mcMatchLayer0.electronMatch_cfi')
    process.merge_step += process.electronMatch
    process.load('PhysicsTools.PatAlgos.mcMatchLayer0.photonMatch_cfi')
    process.merge_step += process.photonMatch
    process.load('PhysicsTools.PatAlgos.mcMatchLayer0.tauMatch_cfi')
    process.merge_step += process.tauMatch
    process.load('PhysicsTools.JetMCAlgos.TauGenJets_cfi')
    process.merge_step += process.tauGenJets
    process.load('PhysicsTools.PatAlgos.mcMatchLayer0.jetFlavourId_cff')
    process.merge_step += process.patJetPartons
    process.load('PhysicsTools.PatAlgos.mcMatchLayer0.jetMatch_cfi')
    process.merge_step += process.patJetPartonMatch

    process.muonMatch.matched = "prunedGenParticles"
    process.electronMatch.matched = "prunedGenParticles"
    process.electronMatch.src = cms.InputTag("reducedEgamma","reducedGedGsfElectrons")
    process.photonMatch.matched = "prunedGenParticles"
    process.photonMatch.src = cms.InputTag("reducedEgamma","reducedGedPhotons")
    process.tauMatch.matched = "prunedGenParticles"
    process.tauGenJets.GenParticles = "prunedGenParticles"
    ##Boosted taus
    #process.tauMatchBoosted.matched = "prunedGenParticles"
    #process.tauGenJetsBoosted.GenParticles = "prunedGenParticles"
    process.patJetPartons.particles = "prunedGenParticles"
    process.patJetPartonMatch.matched = "prunedGenParticles"
    process.patJetPartonMatch.mcStatus = [ 3, 23 ]
    process.patJetGenJetMatch.matched = "slimmedGenJets"
    process.patJetGenJetMatchAK8.matched =  "slimmedGenJetsAK8"
    process.patJetGenJetMatchAK8Puppi.matched =  "slimmedGenJetsAK8"
    process.patMuons.embedGenMatch = False
    process.patElectrons.embedGenMatch = False
    process.patPhotons.embedGenMatch = False
    process.patTaus.embedGenMatch = False
    process.patTausBoosted.embedGenMatch = False
    process.patJets.embedGenPartonMatch = False
    #also jet flavour must be switched
    process.patJetFlavourAssociation.rParam = 0.4

    process.schedule.insert(0,process.merge_step)
    process = customisoptions(process)  
    return modify_outputModules(process, [keepMerged(dataTier)])


def customiseMerging(process, changeProcessname=True,reselect=False):
    if changeProcessname:
        process._Process__name = "MERGE"
    if reselect:
        dataTier="RESELECT"
    else:
        dataTier="SELECT"


    process.source.inputCommands = cms.untracked.vstring()
    process.source.inputCommands.append("keep *_*_*_*")

    #process.source.inputCommands.append("drop *_*_*_SELECT")
    #process.source.inputCommands.append("drop *_*_*_SIMembedding")
    #process.source.inputCommands.append("drop *_*_*_LHEembeddingCLEAN")
    #process.source.inputCommands.extend(keepSimulated())
    #process.source.inputCommands.extend(keepCleaned())

    process.load('Configuration.StandardSequences.Reconstruction_Data_cff')
    process.merge_step = cms.Path()


    for akt_manimod in to_bemanipulate:
        if "MERGE" in akt_manimod.steps:
    #if akt_manimod.module_name != 'particleFlowTmp':
    #  continue
            print(akt_manimod.module_name)
            mergCollections_in = cms.VInputTag()
            for instance in akt_manimod.instance:
                mergCollections_in.append(cms.InputTag(akt_manimod.merge_prefix+akt_manimod.module_name,instance,"SIMembedding"))
                mergCollections_in.append(cms.InputTag(akt_manimod.merge_prefix+akt_manimod.module_name,instance,"LHEembeddingCLEAN"))##  Mayb make some process history magic which finds out if it was CLEAN or LHEembeddingCLEAN step
            setattr(process, akt_manimod.module_name, cms.EDProducer(akt_manimod.merger_name,
                                                     mergCollections = mergCollections_in
                                                     )
            )
            process.merge_step +=getattr(process, akt_manimod.module_name)


    process.merge_step += process.doAlldEdXEstimators
    process.merge_step += process.vertexreco
    process.unsortedOfflinePrimaryVertices.beamSpotLabel = cms.InputTag("offlineBeamSpot","",dataTier)
    process.ak4CaloJetsForTrk.srcPVs = cms.InputTag("firstStepPrimaryVertices","",dataTier)

    process.muons.FillDetectorBasedIsolation = cms.bool(False)
    process.muons.FillSelectorMaps = cms.bool(False)
    process.muons.FillShoweringInfo = cms.bool(False)
    process.muons.FillCosmicsIdMap = cms.bool(False)

    process.muonsFromCosmics.fillShowerDigis = cms.bool(False)
    process.muonsFromCosmics1Leg.fillShowerDigis = cms.bool(False)

    process.merge_step += process.highlevelreco

    #process.merge_step.remove(process.reducedEcalRecHitsEE)
    #process.merge_step.remove(process.reducedEcalRecHitsEB)

    process.merge_step.remove(process.ak4JetTracksAssociatorExplicit)

    process.merge_step.remove(process.pfTrack)
    process.merge_step.remove(process.pfConversions)
    process.merge_step.remove(process.pfV0)
    process.merge_step.remove(process.particleFlowDisplacedVertexCandidate)
    process.merge_step.remove(process.particleFlowDisplacedVertex)
    process.merge_step.remove(process.pfDisplacedTrackerVertex)
    process.merge_step.remove(process.pfTrackElec)
    process.merge_step.remove(process.electronsWithPresel)
    process.merge_step.remove(process.mvaElectrons)
    process.merge_step.remove(process.particleFlowBlock)
    process.merge_step.remove(process.particleFlowEGamma)
    process.merge_step.remove(process.gedGsfElectronCores)
    #  process.merge_step.remove(process.gedGsfElectronsTmp)
    process.merge_step.remove(process.gedPhotonCore)
    process.merge_step.remove(process.ecalDrivenGsfElectronCores)
    process.merge_step.remove(process.ecalDrivenGsfElectrons)
    process.merge_step.remove(process.uncleanedOnlyElectronSeeds)
    process.merge_step.remove(process.uncleanedOnlyAllConversions)
    process.merge_step.remove(process.uncleanedOnlyPfTrack)
    process.merge_step.remove(process.uncleanedOnlyPfTrackElec)
    process.merge_step.remove(process.uncleanedOnlyGsfElectrons)
    process.merge_step.remove(process.uncleanedOnlyElectronCkfTrackCandidates)
    process.merge_step.remove(process.cosmicsVeto)
    process.merge_step.remove(process.cosmicsVetoTrackCandidates)
 #   process.merge_step.remove(process.ecalDrivenGsfElectronCores)
 #   process.merge_step.remove(process.ecalDrivenGsfElectrons)
 #   process.merge_step.remove(process.gedPhotonsTmp)
 #   process.merge_step.remove(process.particleFlowTmp)
    process.merge_step.remove(process.hcalnoise)
    process.merge_step.remove(process.lowPtGsfElectronTask)
    process.merge_step.remove(process.gsfTracksOpenConversions)

    process.load('CommonTools.ParticleFlow.genForPF2PAT_cff')

    process.merge_step += process.genForPF2PATSequence

    process.slimmingTask.remove(process.slimmedLowPtElectronsTask)

    process.schedule.insert(0,process.merge_step)
    # process.load('PhysicsTools.PatAlgos.slimming.slimmedGenJets_cfi')

    process = customisoptions(process) 
    return modify_outputModules(process, [keepMerged(dataTier)])

def customiseMerging_Reselect(process, changeProcessname=True):
    return customiseMerging(process, changeProcessname=changeProcessname, reselect=True)

################################ cross Customizers ###########################

def customiseLHEandCleaning(process,reselect=False):
    process._Process__name = "LHEembeddingCLEAN"
    process = customiseCleaning(process,changeProcessname=False,reselect=reselect)
    process = customiseLHE(process,changeProcessname=False,reselect=reselect)
    return process

def customiseLHEandCleaning_Reselect(process):
    return customiseLHEandCleaning(process,reselect=True)

################################ additionla Customizer ###########################

def customisoptions(process):
    if not hasattr(process, "options"):
        process.options = cms.untracked.PSet()
    process.options.emptyRunLumiMode = cms.untracked.string('doNotHandleEmptyRunsAndLumis')
    if not hasattr(process, "bunchSpacingProducer"):
        process.bunchSpacingProducer = cms.EDProducer("BunchSpacingProducer")
    process.bunchSpacingProducer.bunchSpacingOverride = cms.uint32(25)
    process.bunchSpacingProducer.overrideBunchSpacing = cms.bool(True)
    process.options.numberOfThreads = cms.untracked.uint32(1)
    process.options.numberOfStreams = cms.untracked.uint32(0)
    return process

############################### MC specific Customizer ###########################

def customiseFilterZToMuMu(process):
    process.load("TauAnalysis.MCEmbeddingTools.DYToMuMuGenFilter_cfi")
    process.ZToMuMuFilter = cms.Path(process.dYToMuMuGenFilter)
    process.schedule.insert(-1,process.ZToMuMuFilter)
    return process

def customiseFilterTTbartoMuMu(process):
    process.load("TauAnalysis.MCEmbeddingTools.TTbartoMuMuGenFilter_cfi")
    process.MCFilter = cms.Path(process.TTbartoMuMuGenFilter)
    return customiseMCFilter(process)

def customiseMCFilter(process):
    process.schedule.insert(-1,process.MCFilter)
    outputModulesList = [key for key,value in six.iteritems(process.outputModules)]
    for outputModule in outputModulesList:
        outputModule = getattr(process, outputModule)
        outputModule.SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring("MCFilter"))
    return process

def fix_input_tags(process, formodules = ["generalTracks","cscSegments","dt4DSegments","rpcRecHits"]):
    def change_tags_process(test_input):
        if isinstance(test_input, cms.InputTag):
            if test_input.getModuleLabel() in formodules:
                test_input.setProcessName(process._Process__name)

    def search_for_tags(pset):
        if isinstance(pset, dict):
            for key in pset:
                if isinstance(pset[key], cms.VInputTag):
                    for akt_inputTag in pset[key]:
                        change_tags_process(akt_inputTag)
                elif isinstance(pset[key], cms.PSet):
                    search_for_tags(pset[key].__dict__)
                elif isinstance(pset[key], cms.VPSet):
                    for akt_pset in pset[key]:
                        search_for_tags(akt_pset.__dict__)
                else:
                    change_tags_process(pset[key])
        else:
            print("must be python dict not a ",type(pset))

    for module in process.producers_():
        search_for_tags(getattr(process, module).__dict__)
    for module in process.filters_():
        search_for_tags(getattr(process, module).__dict__)
    for module in process.analyzers_():
        search_for_tags(getattr(process, module).__dict__)

    return process

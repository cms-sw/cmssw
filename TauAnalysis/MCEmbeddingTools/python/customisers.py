#!/usr/bin/env python


### Various set of customise functions needed for embedding
import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.Utilities import cleanUnscheduled


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


to_bemanipulate.append(module_manipulate(module_name = 'dt1DRecHits', manipulator_name = "DTRecHit"))
to_bemanipulate.append(module_manipulate(module_name = 'dt1DCosmicRecHits', manipulator_name = "DTRecHit"))

to_bemanipulate.append(module_manipulate(module_name = 'csc2DRecHits', manipulator_name = "CSCRecHit"))
to_bemanipulate.append(module_manipulate(module_name = 'rpcRecHits', manipulator_name = "RPCRecHit"))


def modify_outputModules(process, keep_drop_list = [], module_veto_list = [] ):
    outputModulesList = [key for key,value in process.outputModules.iteritems()]
    for outputModule in outputModulesList:
	if outputModule in module_veto_list:
	  continue
        outputModule = getattr(process, outputModule)
        for add_element in keep_drop_list:
	  outputModule.outputCommands.extend(add_element)
    return process



################################ Customizer for Selecting ###########################

def keepSelected():
   ret_vstring = cms.untracked.vstring(
             "keep *_patMuonsAfterID_*_SELECT",
             "keep *_slimmedMuons_*_SELECT",
             "keep *_selectedMuonsForEmbedding_*_SELECT",
             "keep recoVertexs_offlineSlimmedPrimaryVertices_*_SELECT",
             "keep *_firstStepPrimaryVertices_*_SELECT",
             "keep *_offlineBeamSpot_*_SELECT")
   for akt_manimod in to_bemanipulate:
      if "CLEAN" in akt_manimod.steps:
	ret_vstring.append("keep *_"+akt_manimod.module_name+"_*_SELECT")
   return ret_vstring



def customiseSelecting(process):
    process._Process__name = "SELECT"

    process.load('TauAnalysis.MCEmbeddingTools.SelectingProcedure_cff')
    process.patMuonsAfterKinCuts.src = cms.InputTag("slimmedMuons","","SELECT")
    process.patMuonsAfterID = process.patMuonsAfterLooseID.clone()

    process.selecting = cms.Path(process.makePatMuonsZmumuSelection)
    process.schedule.insert(-1, process.selecting)

    outputModulesList = [key for key,value in process.outputModules.iteritems()]
    for outputModule in outputModulesList:
        outputModule = getattr(process, outputModule)
        outputModule.SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring("selecting"))
        outputModule.outputCommands.extend(keepSelected())
    return modify_outputModules(process,[keepSelected()])

################################ Customizer for cleaining ###########################
def keepCleaned():
   ret_vstring = cms.untracked.vstring()
   for akt_manimod in to_bemanipulate:
      if "MERGE" in akt_manimod.steps:
        ret_vstring.append("keep *_"+akt_manimod.module_name+"_*_LHEembeddingCLEAN")
        ret_vstring.append("keep *_"+akt_manimod.module_name+"_*_CLEAN")
   ret_vstring.append("keep *_standAloneMuons_*_LHEembeddingCLEAN")
   ret_vstring.append("keep *_glbTrackQual_*_LHEembeddingCLEAN")
   return ret_vstring



def customiseCleaning(process, changeProcessname=True):
    if changeProcessname:
      process._Process__name = "CLEAN"
    ## Needed for the Calo Cleaner, could also be put into a function wich fix the input parameters
    from TrackingTools.TrackAssociator.default_cfi import TrackAssociatorParameterBlock
    TrackAssociatorParameterBlock.TrackAssociatorParameters.CSCSegmentCollectionLabel = cms.InputTag("cscSegments","","SELECT")
    TrackAssociatorParameterBlock.TrackAssociatorParameters.CaloTowerCollectionLabel = cms.InputTag("towerMaker","","SELECT")
    TrackAssociatorParameterBlock.TrackAssociatorParameters.DTRecSegment4DCollectionLabel = cms.InputTag("dt4DSegments","","SELECT")
    TrackAssociatorParameterBlock.TrackAssociatorParameters.EBRecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEB","SELECT")
    TrackAssociatorParameterBlock.TrackAssociatorParameters.EERecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEE","SELECT")
    TrackAssociatorParameterBlock.TrackAssociatorParameters.HBHERecHitCollectionLabel = cms.InputTag("hbhereco","","SELECT")
    TrackAssociatorParameterBlock.TrackAssociatorParameters.HORecHitCollectionLabel = cms.InputTag("horeco","","SELECT")


    MuonImput = cms.InputTag("selectedMuonsForEmbedding","","")  ## This are the muon
    for akt_manimod in to_bemanipulate:
      if "CLEAN" in akt_manimod.steps:
        oldCollections_in = cms.VInputTag()
        for instance in akt_manimod.instance:
	    oldCollections_in.append(cms.InputTag(akt_manimod.module_name,instance,"SELECT"))
        setattr(process, akt_manimod.module_name, cms.EDProducer(akt_manimod.cleaner_name,
                                                                 MuonCollection = MuonImput,
                                                                 TrackAssociatorParameters = TrackAssociatorParameterBlock.TrackAssociatorParameters,
                                                                 oldCollection = oldCollections_in))
    process.ecalPreshowerRecHit.TrackAssociatorParameters.usePreshower = cms.bool(True)

    return modify_outputModules(process,[keepSelected(),keepCleaned()],["MINIAODoutput"])


################################ Customizer for simulaton ###########################
def keepLHE():
    ret_vstring = cms.untracked.vstring()
    ret_vstring.append("keep *_externalLHEProducer_*_LHE")
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
    return ret_vstring




def customiseLHE(process, changeProcessname=True):
    if changeProcessname:
      process._Process__name = "LHEembedding"
    process.load('TauAnalysis.MCEmbeddingTools.EmbeddingLHEProducer_cfi')
    process.lheproduction = cms.Path(process.makeexternalLHEProducer)
    process.schedule.insert(0,process.lheproduction)

    return modify_outputModules(process,[keepSelected(),keepLHE()],["MINIAODoutput"])


def customiseGenerator(process, changeProcessname=True):
    if changeProcessname:
      process._Process__name = "SIMembedding"

    ## here correct the vertex collection
    process.load('TauAnalysis.MCEmbeddingTools.EmbeddingVertexCorrector_cfi')
    process.VtxSmeared = process.VtxCorrectedToInput.clone()
    print "Correcting Vertex in genEvent to one from input. Replaced 'VtxSmeared' with the Corrector."

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
    return modify_outputModules(process,[keepSelected(),keepCleaned(),keepSimulated()],["AODSIMoutput"])


################################ Customizer for merging ###########################
def keepMerged():
    ret_vstring = cms.untracked.vstring()
    ret_vstring.append("drop *_*_*_SELECT")
    ret_vstring.append("keep *_generator_*_SIMembedding")
    return ret_vstring



def customiseMerging(process, changeProcessname=True):
    if changeProcessname:
      process._Process__name = "MERGE"


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
	print akt_manimod.module_name
        mergCollections_in = cms.VInputTag()
        for instance in akt_manimod.instance:
          mergCollections_in.append(cms.InputTag(akt_manimod.merge_prefix+akt_manimod.module_name,instance,"SIMembedding"))
          mergCollections_in.append(cms.InputTag(akt_manimod.merge_prefix+akt_manimod.module_name,instance,"LHEembeddingCLEAN"))##  Mayb make some process history magic which finds out if it was CLEAN or LHEembeddingCLEAN step
	setattr(process, akt_manimod.module_name, cms.EDProducer(akt_manimod.merger_name,
								 mergCollections = mergCollections_in
								 )
	        )
	process.merge_step +=getattr(process, akt_manimod.module_name)


    process.merge_step += process.vertexreco
    process.unsortedOfflinePrimaryVertices.beamSpotLabel = cms.InputTag("offlineBeamSpot","","SELECT")
    process.ak4CaloJetsForTrk.srcPVs = cms.InputTag("firstStepPrimaryVertices","","SELECT")

    process.muons.FillDetectorBasedIsolation = cms.bool(False)
    process.muons.FillSelectorMaps = cms.bool(False)
    process.muons.FillShoweringInfo = cms.bool(False)
    process.muons.FillCosmicsIdMap = cms.bool(False)


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

    process.load('CommonTools.ParticleFlow.genForPF2PAT_cff')
        
    process.merge_step += process.genForPF2PATSequence
    
    process.schedule.insert(0,process.merge_step)
   # process.load('PhysicsTools.PatAlgos.slimming.slimmedGenJets_cfi')
    
    
    return modify_outputModules(process, [keepMerged()])



################################ cross Customizers ###########################

def customiseLHEandCleaning(process):
    process._Process__name = "LHEembeddingCLEAN"
    process = customiseCleaning(process,False)
    process = customiseLHE(process,False)
    return process

################################ additionla Customizer ###########################

def customisoptions(process):
    try:
      process.options.emptyRunLumiMode = cms.untracked.string('doNotHandleEmptyRunsAndLumis')
    except:
      process.options = cms.untracked.PSet(emptyRunLumiMode = cms.untracked.string('doNotHandleEmptyRunsAndLumis'))
    return process

############################### MC specific Customizer ###########################

def customiseFilterZToMuMu(process):
    process.load("TauAnalysis.MCEmbeddingTools.DYToMuMuGenFilter_cfi")
    process.ZToMuMuFilter = cms.Path(process.dYToMuMuGenFilter)
    process.schedule.insert(-1,process.ZToMuMuFilter)
    return process

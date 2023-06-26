#!/usr/bin/env python


# Various set of customise functions needed for embedding
import FWCore.ParameterSet.Config as cms

from PhysicsTools.NanoAOD.common_cff import ExtVar

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


class module_manipulate:
    def __init__(
        self,
        module_name,
        manipulator_name,
        steps=["SELECT", "CLEAN", "SIM", "MERGE"],
        instance=[""],
        merge_prefix="",
    ):
        self.module_name = module_name
        self.manipulator_name = manipulator_name
        self.steps = steps
        self.instance = instance
        self.merger_name = manipulator_name + "ColMerger"
        self.cleaner_name = manipulator_name + "ColCleaner"
        self.merge_prefix = merge_prefix


to_bemanipulate = []


to_bemanipulate.append(
    module_manipulate(
        module_name="siPixelClusters",
        manipulator_name="Pixel",
        steps=["SELECT", "CLEAN"],
    )
)
to_bemanipulate.append(
    module_manipulate(
        module_name="siStripClusters",
        manipulator_name="Strip",
        steps=["SELECT", "CLEAN"],
    )
)

to_bemanipulate.append(
    module_manipulate(
        module_name="generalTracks", manipulator_name="Track", steps=["SIM", "MERGE"]
    )
)
to_bemanipulate.append(
    module_manipulate(
        module_name="cosmicsVetoTracksRaw", manipulator_name="Track", steps=["SIM", "MERGE"]
    )
)
to_bemanipulate.append(
    module_manipulate(
        module_name="electronGsfTracks",
        manipulator_name="GsfTrack",
        steps=["SIM", "MERGE"],
    )
)
to_bemanipulate.append(
    module_manipulate(
        module_name="lowPtGsfEleGsfTracks",
        manipulator_name="GsfTrack",
        steps=["SIM", "MERGE"],
    )
)
to_bemanipulate.append(
    module_manipulate(
        module_name="conversionStepTracks",
        manipulator_name="Track",
        steps=["SIM", "MERGE"],
    )
)
to_bemanipulate.append(
    module_manipulate(
        module_name="displacedTracks",
        manipulator_name="Track",
        steps=["SIM", "MERGE"],
    )
)
to_bemanipulate.append(
    module_manipulate(
        module_name="ckfInOutTracksFromConversions",
        manipulator_name="Track",
        steps=["SIM", "MERGE"],
    )
)
to_bemanipulate.append(
    module_manipulate(
        module_name="ckfOutInTracksFromConversions",
        manipulator_name="Track",
        steps=["SIM", "MERGE"],
    )
)

to_bemanipulate.append(
    module_manipulate(
        module_name="muons1stStep", manipulator_name="Muon", steps=["SIM", "MERGE"]
    )
)
to_bemanipulate.append(
    module_manipulate(
        module_name="displacedMuons1stStep", manipulator_name="Muon", steps=["SIM", "MERGE"]
    )
)
# to_bemanipulate.append(module_manipulate(module_name = 'gedGsfElectronsTmp', manipulator_name = "GsfElectron", steps = ["SIM", "MERGE"]))
# to_bemanipulate.append(module_manipulate(module_name = 'gedPhotonsTmp', manipulator_name = "Photon", steps = ["SIM", "MERGE"]))
to_bemanipulate.append(
    module_manipulate(
        module_name="conversions", manipulator_name="Conversion", steps=["SIM", "MERGE"]
    )
)
to_bemanipulate.append(
    module_manipulate(
        module_name="allConversions",
        manipulator_name="Conversion",
        steps=["SIM", "MERGE"],
    )
)
to_bemanipulate.append(
    module_manipulate(
        module_name="particleFlowTmp",
        manipulator_name="PF",
        steps=["SIM", "MERGE"],
        instance=[
            "",
            "CleanedHF",
            "CleanedCosmicsMuons",
            "CleanedTrackerAndGlobalMuons",
            "CleanedFakeMuons",
            "CleanedPunchThroughMuons",
            "CleanedPunchThroughNeutralHadrons",
            "AddedMuonsAndHadrons",
        ],
    )
)
to_bemanipulate.append(
    module_manipulate(
        module_name="ecalDigis", manipulator_name="EcalSrFlag", steps=["SIM", "MERGE"]
    )
)
to_bemanipulate.append(
    module_manipulate(
        module_name="hcalDigis", manipulator_name="HcalDigi", steps=["SIM", "MERGE"]
    )
)
to_bemanipulate.append(
    module_manipulate(
        module_name="electronMergedSeeds",
        manipulator_name="ElectronSeed",
        steps=["SIM", "MERGE"],
    )
)
to_bemanipulate.append(
    module_manipulate(
        module_name="ecalDrivenElectronSeeds",
        manipulator_name="EcalDrivenElectronSeed",
        steps=["SIM", "MERGE"],
    )
)

to_bemanipulate.append(
    module_manipulate(
        module_name="ecalRecHit",
        manipulator_name="EcalRecHit",
        instance=["EcalRecHitsEB", "EcalRecHitsEE"],
    )
)
to_bemanipulate.append(
    module_manipulate(
        module_name="ecalPreshowerRecHit",
        manipulator_name="EcalRecHit",
        instance=["EcalRecHitsES"],
    )
)

to_bemanipulate.append(
    module_manipulate(module_name="hbheprereco", manipulator_name="HBHERecHit")
)
to_bemanipulate.append(
    module_manipulate(module_name="hbhereco", manipulator_name="HBHERecHit")
)
to_bemanipulate.append(
    module_manipulate(module_name="zdcreco", manipulator_name="ZDCRecHit")
)

to_bemanipulate.append(
    module_manipulate(module_name="horeco", manipulator_name="HORecHit")
)
to_bemanipulate.append(
    module_manipulate(module_name="hfreco", manipulator_name="HFRecHit")
)
to_bemanipulate.append(
    module_manipulate(module_name="castorreco", manipulator_name="CastorRecHit")
)


to_bemanipulate.append(
    module_manipulate(
        module_name="dt1DRecHits",
        manipulator_name="DTRecHit",
        steps=["SELECT", "CLEAN"],
    )
)
to_bemanipulate.append(
    module_manipulate(
        module_name="dt1DCosmicRecHits",
        manipulator_name="DTRecHit",
        steps=["SELECT", "CLEAN"],
    )
)

to_bemanipulate.append(
    module_manipulate(
        module_name="csc2DRecHits",
        manipulator_name="CSCRecHit",
        steps=["SELECT", "CLEAN"],
    )
)
to_bemanipulate.append(
    module_manipulate(
        module_name="rpcRecHits",
        manipulator_name="RPCRecHit",
        steps=["SELECT", "CLEAN"],
    )
)


def modify_outputModules(process, keep_drop_list=[], module_veto_list=[]):
    outputModulesList = [key for key, value in process.outputModules.items()]
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
        # "drop *_*_*_"+dataTier,
        "keep *_patMuonsAfterID_*_" + dataTier,
        "keep *_slimmedMuons_*_" + dataTier,
        "keep *_slimmedMuonTrackExtras_*_" + dataTier,
        "keep *_selectedMuonsForEmbedding_*_" + dataTier,
        "keep recoVertexs_offlineSlimmedPrimaryVertices_*_" + dataTier,
        "keep *_firstStepPrimaryVertices_*_" + dataTier,
        "keep *_offlineBeamSpot_*_" + dataTier,
        "keep *_ecalDrivenElectronSeeds_*_" + dataTier,
    )
    for akt_manimod in to_bemanipulate:
        if "CLEAN" in akt_manimod.steps:
            ret_vstring.append("keep *_" + akt_manimod.module_name + "_*_" + dataTier)
    return ret_vstring


def customiseSelecting(process, reselect=False):
    if reselect:
        process._Process__name = "RESELECT"
        dataTier = "RESELECT"
        # process.source.inputCommands = cms.untracked.vstring("drop *",
        #     "keep *_*_*_LHC",
        #     "keep *_*_*_HLT",
        # )
    else:
        process._Process__name = "SELECT"
        dataTier = "SELECT"

    process.load("TauAnalysis.MCEmbeddingTools.SelectingProcedure_cff")
    process.patMuonsAfterKinCuts.src = cms.InputTag("slimmedMuons", "", dataTier)
    process.selectedMuonsForEmbedding.PuppiMet = cms.InputTag(
        "slimmedMETsPuppi", "", dataTier
    )
    process.selectedMuonsForEmbedding.Met = cms.InputTag("slimmedMETs", "", dataTier)
    process.patMuonsAfterID = process.patMuonsAfterLooseID.clone()

    process.selecting = cms.Path(process.makePatMuonsZmumuSelection)
    process.schedule.insert(-1, process.selecting)

    outputModulesList = [key for key, value in process.outputModules.items()]
    for outputModule in outputModulesList:
        outputModule = getattr(process, outputModule)
        outputModule.SelectEvents = cms.untracked.PSet(
            SelectEvents=cms.vstring("selecting")
        )
        outputModule.outputCommands.extend(keepSelected(dataTier))

    process = customisoptions(process)
    return modify_outputModules(process, [keepSelected(dataTier)])


def customiseSelecting_Reselect(process):
    return customiseSelecting(process, reselect=True)


################################ Customizer for cleaning ###########################


def keepCleaned(dataTier):
    ret_vstring = cms.untracked.vstring(
        # 	 	                 "drop *_*_*_LHEembeddingCLEAN",
        # 	 	                 "drop *_*_*_CLEAN"
        "drop *_*_*_" + dataTier,
        "keep *_patMuonsAfterID_*_" + dataTier,
        "keep *_slimmedMuons_*_" + dataTier,
        # "keep *_slimmedMuonTrackExtras_*_" + dataTier,
        "keep *_selectedMuonsForEmbedding_*_" + dataTier,
        "keep recoVertexs_offlineSlimmedPrimaryVertices_*_" + dataTier,
        "keep *_firstStepPrimaryVertices_*_" + dataTier,
        "keep *_offlineBeamSpot_*_" + dataTier,
        "keep *_l1extraParticles_*_" + dataTier,
        "keep TrajectorySeeds_*_*_*",
        "keep recoElectronSeeds_*_*_*",
        "drop recoIsoDepositedmValueMap_muIsoDepositTk_*_*" ,
        "drop recoIsoDepositedmValueMap_muIsoDepositTkDisplaced_*_*",
        "drop *_muonSimClassifier_*_*",
        # "keep recoPFClusters_*_*_*",
        # "keep recoPFRecHits_*_*_*"
    )

    for akt_manimod in to_bemanipulate:
        if "MERGE" in akt_manimod.steps:
            ret_vstring.append(
                "keep *_" + akt_manimod.module_name + "_*_LHEembeddingCLEAN"
            )
            ret_vstring.append("keep *_" + akt_manimod.module_name + "_*_CLEAN")
    ret_vstring.append("keep *_standAloneMuons_*_LHEembeddingCLEAN")
    ret_vstring.append("keep *_glbTrackQual_*_LHEembeddingCLEAN")
    return ret_vstring


def customiseCleaning(process, changeProcessname=True, reselect=False):
    if changeProcessname:
        process._Process__name = "CLEAN"
    if reselect:
        dataTier = "RESELECT"
    else:
        dataTier = "SELECT"
    ## Needed for the Calo Cleaner, could also be put into a function wich fix the input parameters
    from TrackingTools.TrackAssociator.default_cfi import TrackAssociatorParameterBlock

    TrackAssociatorParameterBlock.TrackAssociatorParameters.CSCSegmentCollectionLabel = cms.InputTag(
        "cscSegments", "", dataTier
    )
    TrackAssociatorParameterBlock.TrackAssociatorParameters.CaloTowerCollectionLabel = (
        cms.InputTag("towerMaker", "", dataTier)
    )
    TrackAssociatorParameterBlock.TrackAssociatorParameters.DTRecSegment4DCollectionLabel = cms.InputTag(
        "dt4DSegments", "", dataTier
    )
    TrackAssociatorParameterBlock.TrackAssociatorParameters.EBRecHitCollectionLabel = (
        cms.InputTag("ecalRecHit", "EcalRecHitsEB", dataTier)
    )
    TrackAssociatorParameterBlock.TrackAssociatorParameters.EERecHitCollectionLabel = (
        cms.InputTag("ecalRecHit", "EcalRecHitsEE", dataTier)
    )
    TrackAssociatorParameterBlock.TrackAssociatorParameters.HBHERecHitCollectionLabel = cms.InputTag(
        "hbhereco", "", dataTier
    )
    TrackAssociatorParameterBlock.TrackAssociatorParameters.HORecHitCollectionLabel = (
        cms.InputTag("horeco", "", dataTier)
    )

    MuonImput = cms.InputTag("selectedMuonsForEmbedding", "", "")  ## This are the muon
    for akt_manimod in to_bemanipulate:
        if "CLEAN" in akt_manimod.steps:
            oldCollections_in = cms.VInputTag()
            for instance in akt_manimod.instance:
                oldCollections_in.append(
                    cms.InputTag(akt_manimod.module_name, instance, dataTier)
                )
            setattr(
                process,
                akt_manimod.module_name,
                cms.EDProducer(
                    akt_manimod.cleaner_name,
                    MuonCollection=MuonImput,
                    TrackAssociatorParameters=TrackAssociatorParameterBlock.TrackAssociatorParameters,
                    oldCollection=oldCollections_in,
                ),
            )
    process.ecalPreshowerRecHit.TrackAssociatorParameters.usePreshower = cms.bool(True)
    process = customisoptions(process)
    return modify_outputModules(
        process, [keepSelected(dataTier), keepCleaned(dataTier)], ["MINIAODoutput"]
    )


################################ Customizer for LHE ###########################


def keepLHE():
    ret_vstring = cms.untracked.vstring()
    ret_vstring.append("keep *_externalLHEProducer_*_LHEembedding")
    ret_vstring.append("keep *_externalLHEProducer_*_LHEembeddingCLEAN")
    return ret_vstring


def customiseLHE(process, changeProcessname=True, reselect=False):
    if reselect:
        dataTier = "RESELECT"
    else:
        dataTier = "SELECT"
    if changeProcessname:
        process._Process__name = "LHEembedding"
    process.load("TauAnalysis.MCEmbeddingTools.EmbeddingLHEProducer_cfi")
    if reselect:
        process.externalLHEProducer.vertices = cms.InputTag(
            "offlineSlimmedPrimaryVertices", "", "RESELECT"
        )
    process.lheproduction = cms.Path(process.makeexternalLHEProducer)
    process.schedule.insert(0, process.lheproduction)

    process = customisoptions(process)
    return modify_outputModules(
        process,
        [keepSelected(dataTier), keepCleaned(dataTier), keepLHE()],
        ["MINIAODoutput"],
    )


def customiseLHEandCleaning(process, reselect=False):
    process._Process__name = "LHEembeddingCLEAN"
    process = customiseCleaning(process, changeProcessname=False, reselect=reselect)
    process = customiseLHE(process, changeProcessname=False, reselect=reselect)
    return process


def customiseLHEandCleaning_Reselect(process):
    return customiseLHEandCleaning(process, reselect=True)


################################ Customizer for simulaton ###########################


def keepSimulated(process, processname="SIMembedding"):
    ret_vstring = cms.untracked.vstring()
    for akt_manimod in to_bemanipulate:
        if "MERGE" in akt_manimod.steps:
            ret_vstring.append(
                "keep *_" + akt_manimod.module_name + "_*_{}".format(processname)
            )
    ret_vstring.append("keep *_genParticles_*_{}".format(processname))
    ret_vstring.append("keep *_standAloneMuons_*_{}".format(processname))
    ret_vstring.append("keep *_glbTrackQual_*_{}".format(processname))
    ret_vstring.append("keep *_generator_*_{}".format(processname))
    ret_vstring.append("keep *_addPileupInfo_*_{}".format(processname))
    ret_vstring.append("keep *_selectedMuonsForEmbedding_*_*")
    ret_vstring.append("keep *_slimmedAddPileupInfo_*_*")
    ret_vstring.append("keep *_embeddingHltPixelVertices_*_*")
    ret_vstring.append("keep *_*_vertexPosition_*")
    ret_vstring.append("keep recoMuons_muonsFromCosmics_*_*")
    ret_vstring.append("keep recoTracks_cosmicMuons1Leg_*_*")
    ret_vstring.append("keep recoMuons_muonsFromCosmics1Leg_*_*")
    ret_vstring.append("keep *_muonDTDigis_*_*")
    ret_vstring.append("keep *_muonCSCDigis_*_*")
    ret_vstring.append("keep TrajectorySeeds_*_*_*")
    ret_vstring.append("keep recoElectronSeeds_*_*_*")
    ret_vstring.append("drop recoIsoDepositedmValueMap_muIsoDepositTk_*_*")
    ret_vstring.append("drop recoIsoDepositedmValueMap_muIsoDepositTkDisplaced_*_*")
    ret_vstring.append("drop *_muonSimClassifier_*_*")

    # for those two steps, the output has to be modified
    # to keep the information from the cleaning step in the output file
    if processname == "SIMembeddingpreHLT" or processname == "SIMembeddingHLT":
        rawreco_commands = set(process.RAWRECOEventContent.outputCommands)
        rawreco_commands_excl = rawreco_commands - set(
            process.RAWSIMEventContent.outputCommands
        )
        for entry in rawreco_commands_excl:
            if (
                processname == "SIMembeddingpreHLT"
                and "muonReducedTrackExtras" in entry
            ):
                continue
            if not any(
                x in entry
                for x in [
                    "TotemTimingLocalTrack",
                    "ForwardProton",
                    "ctppsDiamondLocalTracks",
                ]
            ):
                ret_vstring.append(entry)
    return ret_vstring


def customiseGenerator(process, changeProcessname=True, reselect=False):
    if reselect:
        dataTier = "RESELECT"
    else:
        dataTier = "SELECT"
    if changeProcessname:
        process._Process__name = "SIMembedding"

    ## here correct the vertex collection

    process.load("TauAnalysis.MCEmbeddingTools.EmbeddingVertexCorrector_cfi")
    process.VtxSmeared = process.VtxCorrectedToInput.clone()
    print(
        "Correcting Vertex in genEvent to one from input. Replaced 'VtxSmeared' with the Corrector."
    )
    process.load("TauAnalysis.MCEmbeddingTools.EmbeddingBeamSpotOnline_cfi")
    process.hltOnlineBeamSpot = process.onlineEmbeddingBeamSpotProducer.clone()
    print(
        "Setting online beam spot in HLTSchedule to the one from input data. Replaced 'hltOnlineBeamSpot' with the offline beam spot."
    )

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

    # Replace HLT vertexing with vertex taken from LHE step
    process.load("TauAnalysis.MCEmbeddingTools.EmbeddingHltPixelVerticesProducer_cfi")
    process.hltPixelVertices = process.embeddingHltPixelVertices.clone()
    process.offlinePrimaryVertices = process.embeddingHltPixelVertices.clone()
    process.firstStepPrimaryVerticesUnsorted = process.embeddingHltPixelVertices.clone()
    process.firstStepPrimaryVerticesPreSplitting = (
        process.embeddingHltPixelVertices.clone()
    )

    process = customisoptions(process)
    ##process = fix_input_tags(process)

    return modify_outputModules(
        process,
        [keepSelected(dataTier), keepCleaned(dataTier), keepSimulated(process)],
        ["AODSIMoutput"],
    )


def customiseGenerator_Reselect(process):
    return customiseGenerator(process, reselect=True)


def customiseGenerator_preHLT(process, changeProcessname=True, reselect=False):
    if reselect:
        dataTier = "RESELECT"
    else:
        dataTier = "SELECT"
    if changeProcessname:
        process._Process__name = "SIMembeddingpreHLT"

    ## here correct the vertex collection

    process.load("TauAnalysis.MCEmbeddingTools.EmbeddingVertexCorrector_cfi")
    process.VtxSmeared = process.VtxCorrectedToInput.clone()
    print(
        "Correcting Vertex in genEvent to one from input. Replaced 'VtxSmeared' with the Corrector."
    )

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

    return modify_outputModules(
        process,
        [
            keepSelected(dataTier),
            keepCleaned(dataTier),
            keepSimulated(process, processname="SIMembeddingpreHLT"),
        ],
        ["AODSIMoutput"],
    )


def customiseGenerator_preHLT_Reselect(process):
    return customiseGenerator_preHLT(process, reselect=True)


def customiseGenerator_HLT(process, changeProcessname=True, reselect=False):
    if reselect:
        dataTier = "RESELECT"
    else:
        dataTier = "SELECT"
    if changeProcessname:
        process._Process__name = "SIMembeddingHLT"

    ## here correct the vertex collection
    process.load("TauAnalysis.MCEmbeddingTools.EmbeddingBeamSpotOnline_cfi")
    process.hltOnlineBeamSpot = process.onlineEmbeddingBeamSpotProducer.clone()
    print(
        "Setting online beam spot in HLTSchedule to the one from input data. Replaced 'hltOnlineBeamSpot' with the offline beam spot."
    )

    # Replace HLT vertexing with vertex taken from LHE step
    process.load("TauAnalysis.MCEmbeddingTools.EmbeddingHltPixelVerticesProducer_cfi")
    process.hltPixelVertices = process.embeddingHltPixelVertices.clone()
    process.offlinePrimaryVertices = process.embeddingHltPixelVertices.clone()
    process.firstStepPrimaryVerticesUnsorted = process.embeddingHltPixelVertices.clone()
    process.firstStepPrimaryVerticesPreSplitting = (
        process.embeddingHltPixelVertices.clone()
    )

    process = customisoptions(process)
    ##process = fix_input_tags(process)

    return modify_outputModules(
        process,
        [
            keepSelected(dataTier),
            keepCleaned(dataTier),
            keepLHE(),
            keepSimulated(process, processname="SIMembeddingpreHLT"),
            keepSimulated(process, processname="SIMembeddingHLT"),
        ],
        ["AODSIMoutput"],
    )


def customiseGenerator_HLT_Reselect(process):
    return customiseGenerator_HLT(process, reselect=True)


def customiseGenerator_postHLT(process, changeProcessname=True, reselect=False):
    if reselect:
        dataTier = "RESELECT"
    else:
        dataTier = "SELECT"
    if changeProcessname:
        process._Process__name = "SIMembedding"

    ## here correct the vertex collection

    # process.load('TauAnalysis.MCEmbeddingTools.EmbeddingVertexCorrector_cfi')
    # process.VtxSmeared = process.VtxCorrectedToInput.clone()
    # print "Correcting Vertex in genEvent to one from input. Replaced 'VtxSmeared' with the Corrector."
    # process.load('TauAnalysis.MCEmbeddingTools.EmbeddingBeamSpotOnline_cfi')
    # process.hltOnlineBeamSpot = process.onlineEmbeddingBeamSpotProducer.clone()
    # print "Setting online beam spot in HLTSchedule to the one from input data. Replaced 'hltOnlineBeamSpot' with the offline beam spot."

    # Remove BeamSpot Production, use the one from selected data instead.
    process.reconstruction.remove(process.offlineBeamSpot)

    process = customisoptions(process)
    ##process = fix_input_tags(process)

    return modify_outputModules(
        process,
        [
            keepSelected(dataTier),
            keepCleaned(dataTier),
            keepLHE(),
            keepSimulated(process, processname="SIMembeddingpreHLT"),
            keepSimulated(process, processname="SIMembeddingHLT"),
            keepSimulated(process, processname="SIMembedding"),
        ],
        ["AODSIMoutput"],
    )


def customiseGenerator_postHLT_Reselect(process):
    return customiseGenerator_postHLT(process, reselect=True)


################################ Customizer for merging ###########################


def keepMerged(dataTier="SELECT"):
    ret_vstring = cms.untracked.vstring()
    ret_vstring.append("drop *_*_*_" + dataTier)
    ret_vstring.append("keep *_prunedGenParticles_*_MERGE")
    ret_vstring.append("keep *_generator_*_SIMembeddingpreHLT")
    ret_vstring.append("keep *_generator_*_SIMembeddingHLT")
    ret_vstring.append("keep *_generator_*_SIMembedding")
    ret_vstring.append("keep *_selectedMuonsForEmbedding_*_*")
    ret_vstring.append("keep *_unpackedPatTrigger_*_*")
    ret_vstring.extend(cms.untracked.vstring(
        'keep patPackedGenParticles_packedGenParticles_*_*',
        'keep recoGenParticles_prunedGenParticles_*_*',
        'keep *_packedPFCandidateToGenAssociation_*_*',
        'keep *_lostTracksToGenAssociation_*_*',
        'keep LHEEventProduct_*_*_*',
        'keep GenFilterInfo_*_*_*',
        'keep GenLumiInfoHeader_generator_*_*',
        'keep GenLumiInfoProduct_*_*_*',
        'keep GenEventInfoProduct_generator_*_*',
        'keep recoGenParticles_genPUProtons_*_*',
        'keep *_slimmedGenJetsFlavourInfos_*_*',
        'keep *_slimmedGenJets__*',
        'keep *_slimmedGenJetsAK8__*',
        'keep *_slimmedGenJetsAK8SoftDropSubJets__*',
        'keep *_genMetTrue_*_*',
        # RUN
        'keep LHERunInfoProduct_*_*_*',
        'keep GenRunInfoProduct_*_*_*',
        'keep *_genParticles_xyz0_*',
        'keep *_genParticles_t0_*'))
    return ret_vstring


def customiseKeepPrunedGenParticles(process, reselect=False):
    if reselect:
        dataTier = "RESELECT"
    else:
        dataTier = "SELECT"

    process.keep_step = cms.Path()

    process.load("PhysicsTools.PatAlgos.slimming.genParticles_cff")
    process.keep_step += process.prunedGenParticlesWithStatusOne
    process.load("PhysicsTools.PatAlgos.slimming.prunedGenParticles_cfi")
    process.keep_step += process.prunedGenParticles
    process.load("PhysicsTools.PatAlgos.slimming.packedGenParticles_cfi")
    process.keep_step += process.packedGenParticles
    process.load("PhysicsTools.PatAlgos.slimming.slimmedGenJets_cfi")
    process.keep_step += process.slimmedGenJets

    process.load("PhysicsTools.PatAlgos.mcMatchLayer0.muonMatch_cfi")
    process.keep_step += process.muonMatch
    process.load("PhysicsTools.PatAlgos.mcMatchLayer0.electronMatch_cfi")
    process.keep_step += process.electronMatch
    process.load("PhysicsTools.PatAlgos.mcMatchLayer0.photonMatch_cfi")
    process.keep_step += process.photonMatch
    process.load("PhysicsTools.PatAlgos.mcMatchLayer0.tauMatch_cfi")
    process.keep_step += process.tauMatch
    process.load("PhysicsTools.JetMCAlgos.TauGenJets_cfi")
    process.keep_step += process.tauGenJets
    process.load("PhysicsTools.PatAlgos.mcMatchLayer0.jetFlavourId_cff")
    process.keep_step += process.patJetPartons
    process.load("PhysicsTools.PatAlgos.mcMatchLayer0.jetMatch_cfi")
    process.keep_step += process.patJetPartonMatch

    process.muonMatch.matched = "prunedGenParticles"
    process.electronMatch.matched = "prunedGenParticles"
    process.electronMatch.src = cms.InputTag("reducedEgamma", "reducedGedGsfElectrons")
    process.photonMatch.matched = "prunedGenParticles"
    process.photonMatch.src = cms.InputTag("reducedEgamma", "reducedGedPhotons")
    process.tauMatch.matched = "prunedGenParticles"
    process.tauGenJets.GenParticles = "prunedGenParticles"
    ##Boosted taus
    # process.tauMatchBoosted.matched = "prunedGenParticles"
    # process.tauGenJetsBoosted.GenParticles = "prunedGenParticles"
    process.patJetPartons.particles = "prunedGenParticles"
    process.patJetPartonMatch.matched = "prunedGenParticles"
    process.patJetPartonMatch.mcStatus = [3, 23]
    process.patJetGenJetMatch.matched = "slimmedGenJets"
    process.patJetGenJetMatchAK8.matched = "slimmedGenJetsAK8"
    process.patMuons.embedGenMatch = False
    process.patElectrons.embedGenMatch = False
    process.patPhotons.embedGenMatch = False
    process.patTaus.embedGenMatch = False
    process.patTausBoosted.embedGenMatch = False
    process.patJets.embedGenPartonMatch = False
    # also jet flavour must be switched
    process.patJetFlavourAssociation.rParam = 0.4

    process.schedule.insert(0, process.keep_step)
    process = customisoptions(process)
    return modify_outputModules(process, [keepMerged(dataTier)])


def customiseMerging(process, changeProcessname=True, reselect=False):

    print("**** Attention: overriding behaviour of 'removeMCMatching' ****")

    from PhysicsTools.PatAlgos.slimming.miniAOD_tools import miniAOD_customizeMC
    def performMCMatching(process, names, postfix, outputModules):
        miniAOD_customizeMC(process)

    import PhysicsTools.PatAlgos.tools.coreTools
    PhysicsTools.PatAlgos.tools.coreTools.removeMCMatching = performMCMatching

    if changeProcessname:
        process._Process__name = "MERGE"
    if reselect:
        dataTier = "RESELECT"
    else:
        dataTier = "SELECT"

    process.source.inputCommands = cms.untracked.vstring()
    process.source.inputCommands.append("keep *_*_*_*")
    process.load("PhysicsTools.PatAlgos.slimming.unpackedPatTrigger_cfi")
    process.unpackedPatTrigger.triggerResults = cms.InputTag("TriggerResults::SIMembeddingHLT")

    # process.source.inputCommands.append("drop *_*_*_SELECT")
    # process.source.inputCommands.append("drop *_*_*_SIMembedding")
    # process.source.inputCommands.append("drop *_*_*_LHEembeddingCLEAN")
    # process.source.inputCommands.extend(keepSimulated())
    # process.source.inputCommands.extend(keepCleaned())

    process.load('Configuration.StandardSequences.RawToDigi_cff')
    process.load("Configuration.StandardSequences.Reconstruction_Data_cff")
    process.merge_step = cms.Path()
    # produce local Calo
    process.load("RecoLocalCalo.Configuration.RecoLocalCalo_cff")
    process.merge_step += process.calolocalreco
    #process.merge_step += process.caloglobalreco
    process.merge_step += process.reducedHcalRecHitsSequence

    # produce hcal towers
    process.load("RecoLocalCalo.CaloTowersCreator.calotowermaker_cfi")
    process.merge_step += process.calotowermaker
    process.merge_step += process.towerMaker

    # produce clusters
    process.load("RecoEcal.Configuration.RecoEcal_cff")
    process.merge_step += process.ecalClusters

    # produce PFCluster Collections
    process.load("RecoParticleFlow.PFClusterProducer.particleFlowCluster_cff")
    process.merge_step += process.particleFlowCluster
    process.load(
        "RecoEcal.EgammaClusterProducers.particleFlowSuperClusteringSequence_cff"
    )
    process.merge_step += process.particleFlowSuperClusteringSequence

    # muonEcalDetIds
    process.load("RecoMuon.MuonIdentification.muons1stStep_cfi")
    process.merge_step += process.muonEcalDetIds
    process.merge_step += process.muonShowerInformation

    # muon Isolation sequences
    process.load("RecoMuon.MuonIsolationProducers.muIsolation_cff")
    process.merge_step += process.muIsolation
    process.merge_step += process.muIsolationDisplaced

    # muon ID selection type sequences
    process.load("RecoMuon.MuonIdentification.muonSelectionTypeValueMapProducer_cff")
    process.merge_step += process.muonSelectionTypeSequence

    # displaced muons extras & tracks
    process.load("RecoMuon.MuonIdentification.displacedMuonReducedTrackExtras_cfi")
    process.merge_step += process.displacedMuonReducedTrackExtras

    process.load("RecoMuon.Configuration.MergeDisplacedTrackCollections_cff")
    process.merge_step += process.displacedTracksSequence

    # Other things
    process.merge_step += process.doAlldEdXEstimators
    process.merge_step += process.vertexreco
    process.unsortedOfflinePrimaryVertices.beamSpotLabel = cms.InputTag(
        "offlineBeamSpot", "", dataTier
    )
    process.ak4CaloJetsForTrk.srcPVs = cms.InputTag(
        "firstStepPrimaryVertices", "", dataTier
    )

    # process.muons.FillDetectorBasedIsolation = cms.bool(False)
    # process.muons.FillSelectorMaps = cms.bool(False)
    # process.muons.FillShoweringInfo = cms.bool(False)
    # process.muons.FillCosmicsIdMap = cms.bool(False)

    # process.displacedMuons.FillDetectorBasedIsolation = cms.bool(False)
    # process.displacedMuons.FillSelectorMaps = cms.bool(False)
    # process.displacedMuons.FillShoweringInfo = cms.bool(False)
    # process.displacedMuons.FillCosmicsIdMap = cms.bool(False)

    # seed configuration needed for seedmerger
    #process.load(
    #    "RecoEgamma.EgammaElectronProducers.ecalDrivenElectronSeedsParameters_cff"
    #)
    #process.ecalDrivenElectronSeeds.SeedConfiguration = cms.PSet(
    #    process.ecalDrivenElectronSeedsParameters
    #)

    process.merge_step += process.highlevelreco
    # process.merge_step.remove(process.reducedEcalRecHitsEE)
    # process.merge_step.remove(process.reducedEcalRecHitsEB)

    # process.merge_step.remove(process.ak4JetTracksAssociatorExplicit)

    # process.merge_step.remove(process.cosmicsVeto)
    # process.merge_step.remove(process.cosmicsVetoTrackCandidates)
    # process.merge_step.remove(process.ecalDrivenGsfElectronCores)
    # process.merge_step.remove(process.ecalDrivenGsfElectrons)
    # process.merge_step.remove(process.gedPhotonsTmp)
    # process.merge_step.remove(process.particleFlowTmp)

    # process.merge_step.remove(process.hcalnoise)

    process.load("CommonTools.ParticleFlow.genForPF2PAT_cff")

    # process.muonsFromCosmics.ShowerDigiFillerParameters.dtDigiCollectionLabel = cms.InputTag("simMuonDTDigis")

    process.merge_step += process.genForPF2PATSequence

    # Replace manipulated modules contained in merg_step with Mergers, and
    # put remaining ones into a list to be sorted to avoid deadlocks
    modules_to_be_ordered = {}
    # prepare reco list to determine indices
    reconstruction_modules_list = str(process.RawToDigi).split(",")
    reconstruction_modules_list += str(process.reconstruction).split(",")
    for akt_manimod in to_bemanipulate:
        if "MERGE" in akt_manimod.steps:
            mergCollections_in = cms.VInputTag()
            for instance in akt_manimod.instance:
                mergCollections_in.append(
                    cms.InputTag(
                        akt_manimod.merge_prefix + akt_manimod.module_name,
                        instance,
                        "SIMembedding",
                    )
                )
                mergCollections_in.append(
                    cms.InputTag(
                        akt_manimod.merge_prefix + akt_manimod.module_name,
                        instance,
                        "LHEembeddingCLEAN",
                    )
                )
            setattr(
                process,
                akt_manimod.module_name,
                cms.EDProducer(
                    akt_manimod.merger_name, mergCollections=mergCollections_in
                ),
            )
            if not process.merge_step.contains(getattr(process, akt_manimod.module_name)):
                modules_to_be_ordered[akt_manimod.module_name] = -1
    # Determine indices and place them in right order into the list
    for name,index in modules_to_be_ordered.items():
        if name in reconstruction_modules_list:
            modules_to_be_ordered[name] = reconstruction_modules_list.index(name)
        else:
            print("ERROR:",name,"not prepared in modules list. Please adapt 'customiseMerging'")
            sys.exit(1)

    modules_ordered = sorted(list(modules_to_be_ordered.items()), key=lambda x : -x[1])
    for m in modules_ordered:
        process.merge_step.insert(0, getattr(process, m[0]))


    process.schedule.insert(0, process.merge_step)
    # process.load('PhysicsTools.PatAlgos.slimming.slimmedGenJets_cfi')
    process = customisoptions(process)
    return modify_outputModules(process, [keepMerged(dataTier)])


def customiseMerging_Reselect(process, changeProcessname=True):
    return customiseMerging(process, changeProcessname=changeProcessname, reselect=True)


################################ Customize NanoAOD ################################


def customiseNanoAOD(process):

    process.load("PhysicsTools.NanoAOD.nano_cff")
    process.nanoAOD_step.insert(0, cms.Sequence(process.nanoTableTaskFS))


    for outputModule in process.outputModules.values():
       outputModule.outputCommands.append("keep edmTriggerResults_*_*_SIMembeddingHLT")
       outputModule.outputCommands.append("keep edmTriggerResults_*_*_SIMembedding")
       outputModule.outputCommands.append("keep edmTriggerResults_*_*_MERGE")
       outputModule.outputCommands.append("keep edmTriggerResults_*_*_NANO")
       outputModule.outputCommands.remove("keep edmTriggerResults_*_*_*")

    process.embeddingTable = cms.EDProducer(
        "GlobalVariablesTableProducer",
        name=cms.string("TauEmbedding"),
        # doc=cms.string("TauEmbedding"),
        variables=cms.PSet(
            nInitialPairCandidates=ExtVar(
                cms.InputTag("selectedMuonsForEmbedding", "nPairCandidates"),
                float,
                doc="number of muons pairs suitable for selection (for internal studies only)",
            ),
            SelectionOldMass=ExtVar(
                cms.InputTag("selectedMuonsForEmbedding", "oldMass"),
                float,
                doc="Mass of the Dimuon pair using the old selection algorithm (for internal studies only)",
            ),
            SelectionNewMass=ExtVar(
                cms.InputTag("selectedMuonsForEmbedding", "newMass"),
                float,
                doc="Mass of the Dimuon pair using the new selection algorithm (for internal studies only)",
            ),
            isMediumLeadingMuon=ExtVar(
                cms.InputTag("selectedMuonsForEmbedding", "isMediumLeadingMuon"),
                bool,
                doc="leading muon ID (medium)",
            ),
            isMediumTrailingMuon=ExtVar(
                cms.InputTag("selectedMuonsForEmbedding", "isMediumTrailingMuon"),
                bool,
                doc="trailing muon ID (medium)",
            ),
            isTightLeadingMuon=ExtVar(
                cms.InputTag("selectedMuonsForEmbedding", "isTightLeadingMuon"),
                bool,
                doc="leading muon ID (tight)",
            ),
            isTightTrailingMuon=ExtVar(
                cms.InputTag("selectedMuonsForEmbedding", "isTightTrailingMuon"),
                bool,
                doc="trailing muon ID (tight)",
            ),
            initialMETEt=ExtVar(
                cms.InputTag("selectedMuonsForEmbedding", "initialMETEt"),
                float,
                doc="MET Et of selected event",
            ),
            initialMETphi=ExtVar(
                cms.InputTag("selectedMuonsForEmbedding", "initialMETphi"),
                float,
                doc="MET phi of selected event",
            ),
            initialPuppiMETEt=ExtVar(
                cms.InputTag("selectedMuonsForEmbedding", "initialPuppiMETEt"),
                float,
                doc="PuppiMET Et of selected event",
            ),
            initialPuppiMETphi=ExtVar(
                cms.InputTag("selectedMuonsForEmbedding", "initialPuppiMETphi"),
                float,
                doc="PuppiMET phi of selected event",
            ),
        ),
    )
    process.embeddingTableTask = cms.Task(process.embeddingTable)
    process.schedule.associate(process.embeddingTableTask)

    return process


################################ cross Customizers ###########################


################################ additional Customizer ###########################


def customisoptions(process):
    if not hasattr(process, "options"):
        process.options = cms.untracked.PSet()
    process.options.emptyRunLumiMode = cms.untracked.string(
        "doNotHandleEmptyRunsAndLumis"
    )
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
    process.schedule.insert(-1, process.ZToMuMuFilter)
    return process


def customiseFilterTTbartoMuMu(process):
    process.load("TauAnalysis.MCEmbeddingTools.TTbartoMuMuGenFilter_cfi")
    process.MCFilter = cms.Path(process.TTbartoMuMuGenFilter)
    return customiseMCFilter(process)


def customiseMCFilter(process):
    process.schedule.insert(-1, process.MCFilter)
    outputModulesList = [key for key, value in process.outputModules.items()]
    for outputModule in outputModulesList:
        outputModule = getattr(process, outputModule)
        outputModule.SelectEvents = cms.untracked.PSet(
            SelectEvents=cms.vstring("MCFilter")
        )
    return process


def fix_input_tags(
    process, formodules=["generalTracks", "cscSegments", "dt4DSegments", "rpcRecHits"]
):
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
            print("must be python dict not a {}".format(type(pset)))

    for module in process.producers_():
        search_for_tags(getattr(process, module).__dict__)
    for module in process.filters_():
        search_for_tags(getattr(process, module).__dict__)
    for module in process.analyzers_():
        search_for_tags(getattr(process, module).__dict__)

    return process

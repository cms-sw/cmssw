""" combined_cmssw.py
cmsRun config file to be used with the CmsswPreprocessor for Heppy-Ntupelizing.
Schedules:
 - b-tagging
 - boosted variables
"""

########################################
# Imports/Setup
########################################

import sys
import FWCore.ParameterSet.Config as cms

# Jet Clustering Defaults
from RecoJets.JetProducers.AnomalousCellParameters_cfi import *
from RecoJets.JetProducers.PFJetParameters_cfi import *

# B-Tagging
from RecoBTag.SoftLepton.softPFMuonTagInfos_cfi import *
from RecoBTag.SoftLepton.softPFElectronTagInfos_cfi import *
from RecoBTag.SecondaryVertex.trackSelection_cff import *
from Configuration.AlCa.GlobalTag import GlobalTag

# This function is called by the cmsswPreprocessor 
# (has to be named initialize, can have arbitrary arguments as long as
# all have a default value)
def initialize(**kwargs):
    isMC = kwargs.get("isMC", True)
    lumisToProcess = kwargs.get("lumisToProcess", None)

    process = cms.Process("EX")
    if lumisToProcess == None:
        process.source = cms.Source("PoolSource",
            fileNames = cms.untracked.vstring("file:///scratch/gregor/TTJets_MSDecaysCKM_central_Tune4C_13TeV_MiniAOD.root")
        )
    else:
        process.source = cms.Source("PoolSource",
            fileNames = cms.untracked.vstring("file:///scratch/gregor/TTJets_MSDecaysCKM_central_Tune4C_13TeV_MiniAOD.root"),
            lumisToProcess = lumisToProcess
        )
    process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )

    process.OUT = cms.OutputModule("PoolOutputModule",
        fileName = cms.untracked.string('test.root'),
        outputCommands = cms.untracked.vstring(['drop *'])
    )
    process.endpath= cms.EndPath(process.OUT)

    # Let CMSSW take care of scheduling 
    process.options = cms.untracked.PSet(     
        wantSummary = cms.untracked.bool(True),
        allowUnscheduled = cms.untracked.bool(True)
    )

    skip_ca15 = False

    # 76X PU ID
#    process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#    process.GlobalTag = GlobalTag(process.GlobalTag, '76X_mcRun2_asymptotic_RunIIFall15DR76_v1')
    process.load("RecoJets.JetProducers.PileupJetID_cfi")
    process.pileupJetIdUpdated = process.pileupJetId.clone(
    jets=cms.InputTag("slimmedJets"),
      inputIsCorrected=True,
      applyJec=False,
      vertexes=cms.InputTag("offlineSlimmedPrimaryVertices")
    )
    process.OUT.outputCommands.append("keep *_pileupJetIdUpdated_fullId_EX")

    ########################################
    # Boosted Substructure
    ########################################


    # Select candidates that would pass CHS requirements
    process.chs = cms.EDFilter("CandPtrSelector", src = cms.InputTag("packedPFCandidates"), cut = cms.string("fromPV"))

    if isMC:
        process.OUT.outputCommands.append("keep *_slimmedJetsAK8_*_PAT")
    else:
        process.OUT.outputCommands.append("keep *_slimmedJetsAK8_*_RECO")

    if not skip_ca15:
        # CA, R=1.5, pT > 200 GeV
        process.ca15PFJetsCHS = cms.EDProducer(
                "FastjetJetProducer",
                PFJetParameters,
                AnomalousCellParameters,
                jetAlgorithm = cms.string("CambridgeAachen"),
                rParam       = cms.double(1.5))
        process.ca15PFJetsCHS.src = cms.InputTag("chs")
        process.ca15PFJetsCHS.jetPtMin = cms.double(200.)

        # Calculate tau1, tau2 and tau3 for ungroomed CA R=1.5 jets
        process.ca15PFJetsCHSNSubjettiness  = cms.EDProducer("NjettinessAdder",
                                                             src=cms.InputTag("ca15PFJetsCHS"),
                                                             cone=cms.double(1.5),
                                                             Njets = cms.vuint32(1,2,3),
                                                             # variables for measure definition : 
                                                             measureDefinition = cms.uint32( 0 ), # CMS default is normalized measure
                                                             beta = cms.double(1.0),              # CMS default is 1
                                                             R0 = cms.double(1.5),                # CMS default is jet cone size
                                                             Rcutoff = cms.double( 999.0),       # not used by default
                                                             # variables for axes definition :
                                                             axesDefinition = cms.uint32( 6 ),    # CMS default is 1-pass KT axes
                                                             nPass = cms.int32(999),             # not used by default
                                                             akAxesR0 = cms.double(-999.0)        # not used by default
        )

        # Apply pruning to CA R=1.5 jets
        process.ca15PFPrunedJetsCHS = process.ca15PFJetsCHS.clone(
            usePruning = cms.bool(True),
            nFilt = cms.int32(2),
            zcut = cms.double(0.1),
            rcut_factor = cms.double(0.5),
            useExplicitGhosts = cms.bool(True),
            writeCompound = cms.bool(True), # Also write subjets for pruned fj
            jetCollInstanceName=cms.string("SubJets"),
        )

        # Apply softdrop to CA R=1.5 jets
        process.ca15PFSoftdropJetsCHS = process.ca15PFJetsCHS.clone(
            useSoftDrop = cms.bool(True),
            zcut = cms.double(0.1),
            beta = cms.double(0.0),
            R0 = cms.double(1.5),
            useExplicitGhosts = cms.bool(True))

        # Apply softdrop z=0.2, beta=1 to CA R=1.5 jets
        process.ca15PFSoftdropZ2B1JetsCHS = process.ca15PFJetsCHS.clone(
            useSoftDrop = cms.bool(True),
            zcut = cms.double(0.2),
            beta = cms.double(1.),
            R0 = cms.double(1.5),
            useExplicitGhosts = cms.bool(True))

        # Apply trimming to CA R=1.5 jets
        process.ca15PFTrimmedJetsCHS = process.ca15PFJetsCHS.clone(
            useTrimming = cms.bool(True),
            rFilt = cms.double(0.2),
            trimPtFracMin = cms.double(0.06),
            useExplicitGhosts = cms.bool(True))

        # Calculate tau1, tau2 and tau3 for softdrop (z=0.2, beta=1) CA R=1.5 jets
        process.ca15PFSoftdropZ2B1JetsCHSNSubjettiness  = cms.EDProducer("NjettinessAdder",
                                                                         src=cms.InputTag("ca15PFSoftdropZ2B1JetsCHS"),
                                                                         cone=cms.double(1.5),
                                                                         Njets = cms.vuint32(1,2,3),
                                                                         # variables for measure definition : 
                                                                         measureDefinition = cms.uint32( 0 ), # CMS default is normalized measure
                                                                         beta = cms.double(1.0),              # CMS default is 1
                                                                         R0 = cms.double(1.5),                # CMS default is jet cone size
                                                                         Rcutoff = cms.double( 999.0),       # not used by default
                                                                         # variables for axes definition :
                                                                         axesDefinition = cms.uint32( 6 ),    # CMS default is 1-pass KT axes
                                                                         nPass = cms.int32(999),             # not used by default
                                                                         akAxesR0 = cms.double(-999.0)        # not used by default
        )


        # HEPTopTagger (MultiR)
        process.looseOptRHTT = cms.EDProducer(
            "HTTTopJetProducer",
            PFJetParameters,
            AnomalousCellParameters,
            jetCollInstanceName=cms.string("SubJets"),
            useExplicitGhosts = cms.bool(True),
            writeCompound  = cms.bool(True), 
            optimalR       = cms.bool(True),
            algorithm      = cms.int32(1),
            jetAlgorithm   = cms.string("CambridgeAachen"),
            rParam         = cms.double(1.5),
            mode           = cms.int32(4),
            minFatjetPt    = cms.double(200.),
            minCandPt      = cms.double(200.),
            minSubjetPt    = cms.double(30.),
            minCandMass    = cms.double(0.),
            maxCandMass    = cms.double(1000),
            massRatioWidth = cms.double(100.),
            minM23Cut      = cms.double(0.),
            minM13Cut      = cms.double(0.),
            maxM13Cut      = cms.double(2.))
        process.looseOptRHTT.src = cms.InputTag("chs")
        process.looseOptRHTT.jetPtMin = cms.double(200.)

        process.OUT.outputCommands.append("keep *_ca15PFJetsCHS_*_EX")
        process.OUT.outputCommands.append("keep *_ca15PFPrunedJetsCHS_*_EX")
        process.OUT.outputCommands.append("keep *_ca15PFSoftdropJetsCHS_*_EX")
        process.OUT.outputCommands.append("keep *_ca15PFSoftdropZ2B1JetsCHS_*_EX")
        process.OUT.outputCommands.append("keep *_ca15PFTrimmedJetsCHS_*_EX")
        process.OUT.outputCommands.append("keep *_ca15PFJetsCHSNSubjettiness_*_EX")
        process.OUT.outputCommands.append("keep *_ca15PFSoftdropZ2B1JetsCHSNSubjettiness_*_EX")
        process.OUT.outputCommands.append("keep *_looseOptRHTT_*_EX")




    ########################################
    # Hbb Tagging
    ########################################

    process.load("Configuration.StandardSequences.MagneticField_cff")
    process.load('Configuration.Geometry.GeometryRecoDB_cff')
    process.load("RecoBTag.Configuration.RecoBTag_cff") # this loads all available b-taggers

    process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
    from Configuration.AlCa.GlobalTag import GlobalTag
    process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc')

    for fatjet_name in ["slimmedJetsAK8", "ca15PFJetsCHS"]:

        if skip_ca15 and (fatjet_name in ["ca15PFJetsCHS"]):
            continue


        if fatjet_name == "slimmedJetsAK8":        
            delta_r = 0.8
            maxSVDeltaRToJet = 0.7
            weightFile = 'RecoBTag/SecondaryVertex/data/BoostedDoubleSV_AK8_BDT_v2.weights.xml.gz'
            jetAlgo = "AntiKt"
        elif fatjet_name == "ca15PFJetsCHS":        
            delta_r = 1.5
            maxSVDeltaRToJet = 1.3
            weightFile = 'RecoBTag/SecondaryVertex/data/BoostedDoubleSV_CA15_BDT_v2.weights.xml.gz'
            jetAlgo = "CambridgeAachen"
        else:
            print "Invalid fatjet for b-tagging: ", fatjet_name
            sys.exit()

        # Define the module names
        impact_info_name          = fatjet_name + "ImpactParameterTagInfos"
        isv_info_name             = fatjet_name + "pfInclusiveSecondaryVertexFinderTagInfos"        
        sm_info_name              = fatjet_name + "softPFMuonsTagInfos"
        se_info_name              = fatjet_name + "softPFElectronsTagInfos"
        bb_comp_name              = fatjet_name + "candidateBoostedDoubleSecondaryVertexComputer"
        tag_name                  = fatjet_name + "pfBoostedDoubleSecondaryVertexBJetTags"

        # Setup the modules
        # IMPACT PARAMETER
        setattr(process, 
                impact_info_name, 
                process.pfImpactParameterTagInfos.clone(
                    primaryVertex = cms.InputTag("offlineSlimmedPrimaryVertices"),
                    candidates = cms.InputTag("packedPFCandidates"),
                    computeProbabilities = cms.bool(False),
                    computeGhostTrack = cms.bool(False),
                    maxDeltaR = cms.double(delta_r),
                    jets = cms.InputTag(fatjet_name),
                ))
        getattr(process, impact_info_name).explicitJTA = cms.bool(False)

        # ISV
        setattr(process,
                isv_info_name,                
                process.pfInclusiveSecondaryVertexFinderTagInfos.clone(
                   extSVCollection               = cms.InputTag('slimmedSecondaryVertices'),
                   trackIPTagInfos               = cms.InputTag(impact_info_name),                
                ))
        getattr(process, isv_info_name).useSVClustering = cms.bool(False)
        getattr(process, isv_info_name).rParam = cms.double(delta_r)
        getattr(process, isv_info_name).extSVDeltaRToJet = cms.double(delta_r)
        getattr(process, isv_info_name).trackSelection.jetDeltaRMax = cms.double(delta_r)
        getattr(process, isv_info_name).vertexCuts.maxDeltaRToJetAxis = cms.double(delta_r)
        getattr(process, isv_info_name).jetAlgorithm = cms.string(jetAlgo)

        # DOUBLE B COMPUTER
        setattr(process,
                bb_comp_name,                
                cms.ESProducer("CandidateBoostedDoubleSecondaryVertexESProducer",
                               trackSelectionBlock,
                               beta = cms.double(1.0),
                               R0 = cms.double(delta_r),
                               maxSVDeltaRToJet = cms.double(maxSVDeltaRToJet),
                               useCondDB = cms.bool(False),
                               weightFile = cms.FileInPath(weightFile),
                               useGBRForest = cms.bool(True),
                               useAdaBoost = cms.bool(False),
                               trackPairV0Filter = cms.PSet(k0sMassWindow = cms.double(0.03))
                           ))
        getattr(process, bb_comp_name).trackSelection.jetDeltaRMax = cms.double(delta_r)

        # TAGS
        setattr(process,
                tag_name, 
                cms.EDProducer("JetTagProducer",
                               jetTagComputer = cms.string(bb_comp_name),
                               tagInfos = cms.VInputTag(cms.InputTag(impact_info_name),
                                                        cms.InputTag(isv_info_name)
                                                    )))


        # SOFT MUON
        setattr(process,
                sm_info_name,
                softPFMuonsTagInfos.clone(
                    jets = cms.InputTag(fatjet_name),
                    muons = cms.InputTag("slimmedMuons"),
                    primaryVertex = cms.InputTag("offlineSlimmedPrimaryVertices")             
                ))

        # SOFT ELECTRON
        setattr(process,
                se_info_name,
                softPFElectronsTagInfos.clone(
                    jets = cms.InputTag(fatjet_name),
                    electrons = cms.InputTag("slimmedElectrons"),
                    primaryVertex = cms.InputTag("offlineSlimmedPrimaryVertices"),                
                    DeltaRElectronJet=cms.double(delta_r),
                ))



        # Produce the output
        for object_name in [impact_info_name, isv_info_name,
                            sm_info_name, se_info_name,          
                            bb_comp_name, tag_name]:

            process.OUT.outputCommands.append("keep *_{0}_*_EX".format(object_name))


    ########################################
    # Subjet b-tagging
    ########################################


    for fatjet_name in ["ca15PFPrunedJetsCHS", "looseOptRHTT"]:

        if skip_ca15 and (fatjet_name in ["ca15PFPrunedJetsCHS", "looseOptRHTT"]):
            continue

        if fatjet_name == "ca15PFPrunedJetsCHS":        
            delta_r = 1.5
            jetAlgo = "CambridgeAachen"
        elif fatjet_name == "looseOptRHTT":
            delta_r = 1.5
            jetAlgo = "CambridgeAachen"
        else:
            print "Invalid fatjet for subjet b-tagging: ", fatjet_name
            sys.exit()

        # Define the module names
        impact_info_name          = fatjet_name + "ImpactParameterTagInfos"
        isv_info_name             = fatjet_name + "pfInclusiveSecondaryVertexFinderTagInfos"        
        csvv2_computer_name       = fatjet_name + "combinedSecondaryVertexV2Computer"
        csvv2ivf_name             = fatjet_name + "pfCombinedInclusiveSecondaryVertexV2BJetTags"        

        # Setup the modules
        # IMPACT PARAMETER
        setattr(process, 
                impact_info_name, 
                process.pfImpactParameterTagInfos.clone(
                    primaryVertex = cms.InputTag("offlineSlimmedPrimaryVertices"),
                    candidates = cms.InputTag("packedPFCandidates"),
                    computeProbabilities = cms.bool(False),
                    computeGhostTrack = cms.bool(False),
                    maxDeltaR = cms.double(delta_r),
                    jets = cms.InputTag(fatjet_name, "SubJets"),
                ))
        getattr(process, impact_info_name).explicitJTA = cms.bool(True)

        # ISV
        setattr(process,
                isv_info_name,                
                process.pfInclusiveSecondaryVertexFinderTagInfos.clone(
                   extSVCollection               = cms.InputTag('slimmedSecondaryVertices'),
                   trackIPTagInfos               = cms.InputTag(impact_info_name),                
                ))
        getattr(process, isv_info_name).useSVClustering = cms.bool(True)
        getattr(process, isv_info_name).rParam = cms.double(delta_r)
        getattr(process, isv_info_name).extSVDeltaRToJet = cms.double(delta_r)
        getattr(process, isv_info_name).trackSelection.jetDeltaRMax = cms.double(delta_r)
        getattr(process, isv_info_name).vertexCuts.maxDeltaRToJetAxis = cms.double(delta_r)
        getattr(process, isv_info_name).jetAlgorithm = cms.string(jetAlgo)
        getattr(process, isv_info_name).fatJets  =  cms.InputTag(fatjet_name.replace("looseOptRHTT","ca15PFJetsCHS"))
        getattr(process, isv_info_name).groomedFatJets  =  cms.InputTag(fatjet_name)

        # CSV V2 COMPUTER
        setattr(process,
                csvv2_computer_name,
                process.candidateCombinedSecondaryVertexV2Computer.clone())

        # CSV IVF V2
        setattr(process,
                csvv2ivf_name,
                process.pfCombinedInclusiveSecondaryVertexV2BJetTags.clone(
                    tagInfos = cms.VInputTag(cms.InputTag(impact_info_name),
                                             cms.InputTag(isv_info_name)),
                    jetTagComputer = cms.string(csvv2_computer_name,)
                ))


        # Produce the output
        process.OUT.outputCommands.append("keep *_{0}_*_EX".format(csvv2ivf_name))




    ###
    ### GenHFHadronMatcher
    ###
    if isMC:
        process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

        genParticleCollection = 'prunedGenParticles'

        genJetCollection = "slimmedGenJets"

        from PhysicsTools.JetMCAlgos.AK4PFJetsMCFlavourInfos_cfi import ak4JetFlavourInfos
        process.genJetFlavourInfos = ak4JetFlavourInfos.clone(
           jets = genJetCollection,
        )


        # Ghost particle collection used for Hadron-Jet association
        # MUST use proper input particle collection
        from PhysicsTools.JetMCAlgos.HadronAndPartonSelector_cfi import selectedHadronsAndPartons
        process.selectedHadronsAndPartons = selectedHadronsAndPartons.clone(
            particles = genParticleCollection
        )

        # Input particle collection for matching to gen jets (partons + leptons)
        # MUST use use proper input jet collection: the jets to which hadrons should be associated
        # rParam and jetAlgorithm MUST match those used for jets to be associated with hadrons
        # More details on the tool: https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideBTagMCTools#New_jet_flavour_definition
#        from PhysicsTools.JetMCAlgos.GenHFHadronMatcher_cff import genJetFlavourPlusLeptonInfos
 #       process.genJetFlavourPlusLeptonInfos = genJetFlavourPlusLeptonInfos.clone(
 #           jets = genJetCollection,
 #           rParam = cms.double(0.4),
 #           jetAlgorithm = cms.string("AntiKt")
 #       )


        # Plugin for analysing B hadrons
        # MUST use the same particle collection as in selectedHadronsAndPartons
        from PhysicsTools.JetMCAlgos.GenHFHadronMatcher_cff import matchGenBHadron
        process.matchGenBHadron = matchGenBHadron.clone(
            genParticles = genParticleCollection
        )

        # Plugin for analysing C hadrons
        # MUST use the same particle collection as in selectedHadronsAndPartons
        from PhysicsTools.JetMCAlgos.GenHFHadronMatcher_cff import matchGenCHadron
        process.matchGenCHadron = matchGenCHadron.clone(
            genParticles = genParticleCollection
        )

        process.categorizeGenTtbar = cms.EDProducer("GenTtbarCategorizer",
            # Phase space of additional jets
            genJetPtMin = cms.double(20.),
            genJetAbsEtaMax = cms.double(2.4),
            # Input tags holding information about b/c hadron matching
            genJets = cms.InputTag(genJetCollection),
            genBHadJetIndex = cms.InputTag("matchGenBHadron", "genBHadJetIndex"),
            genBHadFlavour = cms.InputTag("matchGenBHadron", "genBHadFlavour"),
            genBHadFromTopWeakDecay = cms.InputTag("matchGenBHadron", "genBHadFromTopWeakDecay"),
            genBHadPlusMothers = cms.InputTag("matchGenBHadron", "genBHadPlusMothers"),
            genBHadPlusMothersIndices = cms.InputTag("matchGenBHadron", "genBHadPlusMothersIndices"),
            genBHadIndex = cms.InputTag("matchGenBHadron", "genBHadIndex"),
            genBHadLeptonHadronIndex = cms.InputTag("matchGenBHadron", "genBHadLeptonHadronIndex"),
            genBHadLeptonViaTau = cms.InputTag("matchGenBHadron", "genBHadLeptonViaTau"),
            genCHadJetIndex = cms.InputTag("matchGenCHadron", "genCHadJetIndex"),
            genCHadFlavour = cms.InputTag("matchGenCHadron", "genCHadFlavour"),
            genCHadFromTopWeakDecay = cms.InputTag("matchGenCHadron", "genCHadFromTopWeakDecay"),
            genCHadBHadronId = cms.InputTag("matchGenCHadron", "genCHadBHadronId"),
        )

        process.OUT.outputCommands.append("keep *_matchGenBHadron__EX")
        process.OUT.outputCommands.append("keep *_matchGenCHadron__EX")
        process.OUT.outputCommands.append("keep *_matchGenBHadron_*_EX")
        process.OUT.outputCommands.append("keep *_matchGenCHadron_*_EX")
        process.OUT.outputCommands.append("keep *_categorizeGenTtbar_*_EX")
        process.OUT.outputCommands.append("keep *_categorizeGenTtbar__EX")

    #Schedule to run soft muon and electron taggers on miniAOD
    process.softPFElectronsTagInfos.jets = cms.InputTag("slimmedJets")
    process.softPFElectronsTagInfos.electrons = cms.InputTag("slimmedElectrons")
    process.softPFElectronsTagInfos.primaryVertex = cms.InputTag("offlineSlimmedPrimaryVertices")
            
    process.softPFMuonsTagInfos.jets = cms.InputTag("slimmedJets")
    process.softPFMuonsTagInfos.muons = cms.InputTag("slimmedMuons")
    process.softPFMuonsTagInfos.primaryVertex = cms.InputTag("offlineSlimmedPrimaryVertices")
        
    process.OUT.outputCommands.append("keep *_softPFElectronsTagInfos_*_*")
    process.OUT.outputCommands.append("keep *_softPFMuonsTagInfos_*_*")
    process.OUT.outputCommands.append("keep *_softPFElectronBJetTags_*_EX")
    process.OUT.outputCommands.append("keep *_softPFMuonBJetTags_*_EX")



    ########################################
    # Generator level hadronic tau decays
    ########################################
    if isMC:
        process.load("PhysicsTools.JetMCAlgos.TauGenJets_cfi")
        process.tauGenJets.GenParticles = cms.InputTag('prunedGenParticles')
        process.load("PhysicsTools.JetMCAlgos.TauGenJetsDecayModeSelectorAllHadrons_cfi")
        process.OUT.outputCommands.append("keep *_tauGenJetsSelectorAllHadrons_*_EX")



    ########################################
    # Electron MVA ID: https://twiki.cern.ch/twiki/bin/viewauth/CMS/MultivariateElectronIdentificationRun2#Recipes_and_implementation
    ########################################
   
    from PhysicsTools.SelectorUtils.tools.vid_id_tools import *
    switchOnVIDElectronIdProducer(process, DataFormat.MiniAOD)
    for eleid in ["RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Spring15_25ns_Trig_V1_cff"]:
        setupAllVIDIdsInModule(process, eleid, setupVIDElectronSelection)
    for eleid in ["RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Spring15_25ns_nonTrig_V1_cff"]:
        setupAllVIDIdsInModule(process, eleid, setupVIDElectronSelection)
    process.OUT.outputCommands.append("keep *_electronMVAValueMapProducer_*_EX")
    process.OUT.outputCommands.append("keep *_egmGsfElectronIDs_*_EX")
    
    return process

# Called directly 
# (luckily for us cmsRun also counts as direct call)
if __name__ == "__main__":
    process = initialize()


import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing
import sys

## Define the process
process = cms.Process("Analyzer")
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True),
    allowUnscheduled = cms.untracked.bool(True),
)

## Set up command line options
options = VarParsing.VarParsing ('standard')
options.register('runOnAOD', True, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.bool, "run on AOD")

## Get and parse command line options
if( hasattr(sys, "argv") ):
    for args in sys.argv :
        arg = args.split(',')
        for val in arg:
             val = val.split('=')
             if(len(val)==2):
                 setattr(options,val[0], val[1])

## Configure message logger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = 'INFO'
process.MessageLogger.cerr.FwkReport.reportEvery = 100

## Define input
if options.runOnAOD:
    process.source = cms.Source("PoolSource",
        fileNames = cms.untracked.vstring(    
            # add your favourite AOD files here
#            '/store/relval/CMSSW_7_2_0_pre7/RelValTTbar_13/GEN-SIM-RECO/PU50ns_PRE_LS172_V12-v1/00000/1267B7ED-2F4E-E411-A0B9-0025905964A6.root',
            '/store/mc/Spring14dr/TTJets_MSDecaysCKM_central_Tune4C_13TeV-madgraph-tauola/AODSIM/PU_S14_POSTLS170_V6-v1/00000/00120F7A-84F5-E311-9FBE-002618943910.root',
#            '/store/mc/Spring14dr/TT_Tune4C_13TeV-pythia8-tauola/AODSIM/Flat20to50_POSTLS170_V5-v1/00000/023E1847-ADDC-E311-91A2-003048FFD754.root',
#            '/store/mc/Spring14dr/TTbarH_HToBB_M-125_13TeV_pythia6/AODSIM/PU20bx25_POSTLS170_V5-v1/00000/1CAB7E58-0BD0-E311-B688-00266CFFBC3C.root',
#            '/store/mc/Spring14dr/TTbarH_M-125_13TeV_amcatnlo-pythia8-tauola/AODSIM/PU20bx25_POSTLS170_V5-v1/00000/0E3D08A9-C610-E411-A862-0025B3E0657E.root',
        ),
        skipEvents = cms.untracked.uint32(0)
    )
else:
    process.source = cms.Source("PoolSource",
        fileNames = cms.untracked.vstring(    
            # add your favourite miniAOD files here
#            '/store/mc/Phys14DR/TTJets_MSDecaysCKM_central_Tune4C_13TeV-madgraph-tauola/MINIAODSIM/PU20bx25_PHYS14_25_V1-v1/00000/FE26BEB8-D575-E411-A13E-00266CF2AE10.root',
#            '/store/mc/RunIISpring15DR74/TTJets_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8/MINIAODSIM/Asympt25ns_MCRUN2_74_V9-v1/00000/FE3EF0E9-9F02-E511-BA6F-549F35AE4FA2.root',
            '/store/mc/RunIISpring15DR74/TTJets_TuneCUETP8M1_13TeV-amcatnloFXFX-scaledown-pythia8/MINIAODSIM/Asympt50ns_MCRUN2_74_V9A-v2/50000/FEA9ADAC-FB0B-E511-9B8C-00266CF9AB9C.root',
#            '/store/mc/RunIISpring15DR74/TT_TuneZ2star_13TeV-powheg-pythia6-tauola/MINIAODSIM/Asympt25ns_MCRUN2_74_V9-v1/50000/C6F89D57-640A-E511-8D41-549F35AD8BC9.root',
#            '/store/mc/RunIISpring15DR74/TT_TuneZ2star_13TeV-powheg-pythia6-tauola/MINIAODSIM/Asympt50ns_MCRUN2_74_V9A-v3/00000/D4BE2E58-CE08-E511-9247-0025907DC9CC.root',
        ),
        skipEvents = cms.untracked.uint32(0)
    )

## Define maximum number of events to loop over
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)

## Set input particle collections to be used by the tools
genParticleCollection = ''
genJetCollection = ''
if options.runOnAOD:
    genParticleCollection = 'genParticles'
    genJetCollection = 'ak4GenJetsCustom'
    ## producing a subset of genParticles to be used for jet clustering in AOD
    from RecoJets.Configuration.GenJetParticles_cff import genParticlesForJetsNoNu
    process.genParticlesForJetsCustom = genParticlesForJetsNoNu.clone()
    ## Produce own jets (re-clustering in miniAOD needed at present to avoid crash)
    from RecoJets.JetProducers.ak4GenJets_cfi import ak4GenJets
    process.ak4GenJetsCustom = ak4GenJets.clone(
        src = 'genParticlesForJetsCustom',
        rParam = cms.double(0.4),
        jetAlgorithm = cms.string("AntiKt")
    )
else:
    genParticleCollection = 'prunedGenParticles'
    genJetCollection = 'slimmedGenJets'

## Supplies PDG ID to real name resolution of MC particles
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")


## Ghost particle collection used for Hadron-Jet association 
# MUST use proper input particle collection
from PhysicsTools.JetMCAlgos.HadronAndPartonSelector_cfi import selectedHadronsAndPartons
process.selectedHadronsAndPartons = selectedHadronsAndPartons.clone(
    particles = genParticleCollection
)

## Input particle collection for matching to gen jets (partons + leptons) 
# MUST use use proper input jet collection: the jets to which hadrons should be associated
# rParam and jetAlgorithm MUST match those used for jets to be associated with hadrons
# More details on the tool: https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideBTagMCTools#New_jet_flavour_definition
from PhysicsTools.JetMCAlgos.AK4PFJetsMCFlavourInfos_cfi import ak4JetFlavourInfos
process.genJetFlavourInfos = ak4JetFlavourInfos.clone(
    jets = genJetCollection,
)


## Plugin for analysing B hadrons
# MUST use the same particle collection as in selectedHadronsAndPartons
from PhysicsTools.JetMCAlgos.GenHFHadronMatcher_cff import matchGenBHadron
process.matchGenBHadron = matchGenBHadron.clone(
    genParticles = genParticleCollection,
    jetFlavourInfos = "genJetFlavourInfos"
)

## Plugin for analysing C hadrons
# MUST use the same particle collection as in selectedHadronsAndPartons
from PhysicsTools.JetMCAlgos.GenHFHadronMatcher_cff import matchGenCHadron
process.matchGenCHadron = matchGenCHadron.clone(
    genParticles = genParticleCollection,
    jetFlavourInfos = "genJetFlavourInfos"
)

## Producer for ttbar categorisation ID
# MUST use same genJetCollection as used for tools above
from TopQuarkAnalysis.TopTools.GenTtbarCategorizer_cfi import categorizeGenTtbar
process.categorizeGenTtbar = categorizeGenTtbar.clone(
    genJetPtMin = 20.,
    genJetAbsEtaMax = 2.4,
    genJets = genJetCollection,
)

## Configure test analyzer
process.testGenTtbarCategories = cms.EDAnalyzer("TestGenTtbarCategories",
    genTtbarId = cms.InputTag("categorizeGenTtbar", "genTtbarId")
)

## Output root file
process.TFileService = cms.Service("TFileService",
    fileName = cms.string("genTtbarId.root")
)

## Path
process.p1 = cms.Path(
    process.testGenTtbarCategories
)


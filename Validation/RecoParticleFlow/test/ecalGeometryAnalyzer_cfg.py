import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing
    
# cmsRun <full_path_to>/ecalGeometryAnalyzer_cfg.py input=step2.root maxEvents=10
opt = VarParsing.VarParsing('analysis')
opt.register(
    'input', '',
    VarParsing.VarParsing.multiplicity.singleton,
    VarParsing.VarParsing.varType.string,
    "Input file (only one supported)"
)
opt.register(
    'output', '',
    VarParsing.VarParsing.multiplicity.singleton,
    VarParsing.VarParsing.varType.string,
    "Output file."
)
opt.register(
    'kinematicCuts', False,
    VarParsing.VarParsing.multiplicity.singleton,
    VarParsing.VarParsing.varType.bool,
    "Whether to apply similar kinematic cuts as done in the PF cluster validation."
)
opt.register(
    'enFracCut', 0.01,
    VarParsing.VarParsing.multiplicity.singleton,
    VarParsing.VarParsing.varType.float,
    """
    Cut on the energy fraction (wrt CaloParticle energy) for each sim cluster.
    Considered only if kinematicCuts = True.
    """
)
opt.register(
    'ptCut', 0.1,
    VarParsing.VarParsing.multiplicity.singleton,
    VarParsing.VarParsing.varType.float,
    """
    Cut on the pT of each sim cluster.
    Considered only if kinematicCuts = True.
    """
)
opt.register(
    'scoreCut', 1.,
    VarParsing.VarParsing.multiplicity.singleton,
    VarParsing.VarParsing.varType.float,
    """
    Cut on the score of the matching between a sim cluster and all reco clusters.
    The score is a distance metric: 0 corresponds to a "perfect" matching, while 1 is the absence of matching.
    Considered only if kinematicCuts = True.
    """
)
opt.register(
    'responseCut', 0.,
    VarParsing.VarParsing.multiplicity.singleton,
    VarParsing.VarParsing.varType.float,
    """
    Cut on the response of each sim cluster wrt to each reco cluster.
    The event is filled only if there is at least one sim/reco combination with a response larger than the cut threshold.
    Always considered.
    """
)
opt.parseArguments()

def noDots(sss):
    return str(sss).replace('.','p')

if opt.output == '':
    if opt.kinematicCuts:
        opt.output = f'data_response{noDots(opt.responseCut)}_pt{noDots(opt.ptCut)}_enfrac{noDots(opt.enFracCut)}_score{noDots(opt.scoreCut)}.root'
    else:
        opt.output = f'data_response{noDots(opt.responseCut)}_nokincut.root'

from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9
from Configuration.ProcessModifiers.enableCPfromPU_cff import enableCPfromPU
process = cms.Process("EcalGeometryAnalyzer",Phase2C17I13M9,enableCPfromPU)

process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.Geometry.GeometryExtendedRun4D121Reco_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load('Configuration.StandardSequences.L1TrackTrigger_cff')
process.load('Configuration.StandardSequences.SimPhase2L1GlobalTriggerEmulator_cff')
process.load('L1Trigger.Configuration.Phase2GTMenus.SeedDefinitions.step1_2024.l1tGTMenu_cff')
process.load('Configuration.StandardSequences.Validation_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic_T33', '') 

process.TFileService = cms.Service(
    "TFileService", 
    fileName = cms.string(opt.output),
    closeFileFast = cms.untracked.bool(True)
)

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1
# process.MessageLogger.cerr.threshold = 'INFO'
# process.MessageLogger.cerr.INFO.limit = -1
# process.MessageLogger.debugModules = ["*"]

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(opt.maxEvents)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:' + opt.input)
)

ecalRecoClusters = "hltParticleFlowClusterECALUnseeded"

process.hltPFScAssocByEnergyScoreProducer = cms.EDProducer("BarrelPCToSCAssociatorByEnergyScoreProducer",
    hardScatterOnly = cms.bool(True),
    hitMapTag = cms.InputTag("hltRecHitMapProducer:barrelRecHitMap"),
    hits = cms.VInputTag("hltParticleFlowRecHitECALUnseeded", "hltParticleFlowRecHitHBHE"),
    # hltParticleFlowClusterHO
)

process.hltPFClusterSimClusterAssociationProducerECAL = cms.EDProducer("PCToSCAssociatorEDProducer",
    associator = cms.InputTag("hltPFScAssocByEnergyScoreProducer"),
    label_lcl = cms.InputTag(ecalRecoClusters),
    label_scl = cms.InputTag("mix","MergedCaloTruth")
)

process.ecalGeometryAnalyzer = cms.EDAnalyzer(
    'EcalGeometryAnalyzer',
    caloParticles = cms.InputTag("mix", "MergedCaloTruth"),
    recHits = cms.InputTag("hltParticleFlowRecHitECALUnseeded"),
    simHits = cms.InputTag("g4SimHits", "EcalHitsEB"),
    recClusters = cms.InputTag(ecalRecoClusters),
    simClusters = cms.InputTag("mix", "MergedCaloTruth"),
    kinematicCuts = cms.untracked.bool(opt.kinematicCuts),
    enFracCut = cms.untracked.double(opt.enFracCut),
    ptCut = cms.untracked.double(opt.ptCut),
    scoreCut = cms.untracked.double(opt.scoreCut),
    responseCut = cms.untracked.double(opt.responseCut),
)

"""
This cut avoids the need to have associator information in the event
if it is not needed
"""
if opt.kinematicCuts or opt.responseCut > 0.:
    process.ecalGeometryAnalyzer.clusterAssociator = cms.InputTag("hltPFClusterSimClusterAssociationProducerECAL")
    process.p = cms.Path(
        process.hltPFScAssocByEnergyScoreProducer
        * process.hltPFClusterSimClusterAssociationProducerECAL
        * process.ecalGeometryAnalyzer
    )
else:
    process.p = cms.Path(process.ecalGeometryAnalyzer)

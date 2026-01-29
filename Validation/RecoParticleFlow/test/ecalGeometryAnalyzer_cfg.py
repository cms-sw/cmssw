import argparse
import FWCore.ParameterSet.Config as cms

def noDots(obj):
    return str(obj).replace('.','p')

class dotDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

#######################################################################################
################ PARSER ###############################################################
#######################################################################################
parser = argparse.ArgumentParser(description='Produce ECAL geometry.')
eraChoices = ['Run3', 'Phase2']

parser.add_argument("-i", "--infile", help='Input file (only one supported).')
parser.add_argument("-o", "--outfile", default='', help='Output file.')
parser.add_argument("-k", "--kinematicCuts", type=bool, default=False,
                    help='Whether to apply similar kinematic cuts as done in the PF cluster validation.')
parser.add_argument("-e", "--enFracCut", type=float, default=0.01,
                    help='Cut on the energy fraction (wrt CaloParticle energy) for each sim cluster. Considered only if kinematicCuts = True.')
parser.add_argument("-p", "--ptCut", type=float, default=0.1,
                    help='Cut on the pT of each sim cluster. Considered only if kinematicCuts = True.')
parser.add_argument("-s", "--scoreCut", type=float, default=1.,
                    help='Cut on the score of the matching between a sim cluster and all reco clusters. The score is a distance metric: 0 corresponds to a "perfect" matching, while 1 is the absence of matching. Considered only if kinematicCuts = True.')
parser.add_argument("-r", "--responseCut", type=float, default=0.,
                    help='Cut on the response of each sim cluster with respect to each reco cluster. The event is filled only if there is at least one sim/reco combination with a response larger than the cut threshold. Always considered.')
parser.add_argument("--era", default='Phase2', choices=eraChoices,
                    help='Era (Run3 or Phase-2).')
parser.add_argument("-n", "--maxEvents", type=int, default=10,
                    help='Cut on the pT of each sim cluster. Considered only if kinematicCuts = True.')
args = parser.parse_args()

if any(x in args.infile and x != args.era for x in eraChoices):
    print('=========== WARNING ===================')
    print('Please make sure the following input arguments make sense:')
    print(f'  --era {args.era}')
    print(f'  --infile {args.infile}')
    print('======================================')
    
if args.outfile == '':
    if args.kinematicCuts:
        args.outfile = f'data_response{noDots(args.responseCut)}_pt{noDots(args.ptCut)}_enfrac{noDots(args.enFracCut)}_score{noDots(args.scoreCut)}.root'
    else:
        args.outfile = f'data_response{noDots(args.responseCut)}_nokincut.root'

cuts = dotDict(kin=args.kinematicCuts, enFrac=args.enFracCut, pt=args.ptCut, score=args.scoreCut,
               response=args.responseCut)
#######################################################################################
#######################################################################################
#######################################################################################

from Configuration.AlCa.GlobalTag import GlobalTag
from Configuration.ProcessModifiers.enableCPfromPU_cff import enableCPfromPU
    
if args.era == 'Phase2':
    from Configuration.Eras.Era_Phase2C22I13M9_cff import Phase2C22I13M9
    era = Phase2C22I13M9
    globalTag = 'auto:phase2_realistic_T33'

elif args.era == 'Run3':
    from Configuration.Eras.Era_Run3_2025_cff import Run3_2025
    era = Run3_2025
    globalTag = 'auto:phase1_2025_realistic'

process = cms.Process("EcalGeometryAnalyzer", era, enableCPfromPU)
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.Validation_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

if args.era == 'Phase2':
    process.load('Configuration.Geometry.GeometryExtendedRun4D121Reco_cff')
elif args.era == 'Run3':
    process.load('Configuration.Geometry.GeometryRecoDB_cff')
    
process.GlobalTag = GlobalTag(process.GlobalTag, globalTag, '')

process.TFileService = cms.Service(
    "TFileService", 
    fileName = cms.string(args.outfile),
    closeFileFast = cms.untracked.bool(True)
)
 
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1
# process.MessageLogger.cerr.threshold = 'INFO'
# process.MessageLogger.cerr.INFO.limit = -1
# process.MessageLogger.debugModules = ["*"]
 
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(args.maxEvents)
)
process.options.numberOfThreads=cms.untracked.uint32(1)
process.options.numberOfStreams=cms.untracked.uint32(1)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:' + args.infile)
)

from RecoLocalCalo.HGCalRecProducers.recHitMapProducer_cff import recHitMapProducer as _recHitMapProducer
process.hltRecHitMapProducer = _recHitMapProducer.clone(
    hits = ["hltParticleFlowRecHitECALUnseeded", "hltParticleFlowRecHitHBHE"],
    doPFHits = True,
    doHgcalHits = False,
)

process.hltPFScAssocByEnergyScoreProducer = cms.EDProducer("BarrelPCToSCAssociatorByEnergyScoreProducer",
    hardScatterOnly = cms.bool(True),
    hitMapTag = cms.InputTag('hltRecHitMapProducer', 'pfRecHitMap'),
    hits = cms.InputTag("hltRecHitMapProducer", "RefProdVectorPFRecHitCollection"),
)
 
process.hltPFClusterSimClusterAssociationProducerECAL = cms.EDProducer("PCToSCAssociatorEDProducer",
    associator = cms.InputTag("hltPFScAssocByEnergyScoreProducer"),
    label_lcl = cms.InputTag("hltParticleFlowClusterECALUnseeded"),
    label_scl = cms.InputTag("mix","MergedCaloTruth")
)

process.ecalGeometryAnalyzer = cms.EDAnalyzer(
    'EcalGeometryAnalyzer',
    caloParticles = cms.InputTag("mix", "MergedCaloTruth"),
    recHits = cms.InputTag("hltParticleFlowRecHitECALUnseeded"),
    simHits = cms.InputTag("g4SimHits", "EcalHitsEB"),
    recClusters = cms.InputTag("hltParticleFlowClusterECALUnseeded"),
    simClusters = cms.InputTag("mix", "MergedCaloTruth"),
    kinematicCuts = cms.untracked.bool(cuts.kin),
    enFracCut = cms.untracked.double(cuts.enFrac),
    ptCut = cms.untracked.double(cuts.pt),
    scoreCut = cms.untracked.double(cuts.score),
    responseCut = cms.untracked.double(cuts.response),
)
 
"""
This cut avoids the need to have associator information in the event
if it is not needed
"""
if cuts.kin or cuts.response > 0.:
    process.ecalGeometryAnalyzer.clusterAssociator = cms.InputTag("hltPFClusterSimClusterAssociationProducerECAL")
    process.p = cms.Path(
        process.hltRecHitMapProducer
        * process.hltPFScAssocByEnergyScoreProducer
        * process.hltPFClusterSimClusterAssociationProducerECAL
        * process.ecalGeometryAnalyzer
    )
else:
    process.p = cms.Path(process.ecalGeometryAnalyzer)


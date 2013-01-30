# Auto generated configuration file
# using: 
# Revision: 1.381.2.2 
# Source: /local/reps/CMSSW/CMSSW/Configuration/PyReleaseValidation/python/ConfigBuilder.py,v 
# with command line options: TauAnalysis/MCEmbeddingTools/python/PFEmbeddingSource_cff -s GEN,SIM,DIGI,L1,DIGI2RAW,HLT:GRun,RAW2DIGI,L1Reco,RECO --no_exec --conditions=FrontierConditions_GlobalTag,START53_V7A::All --fileout=embedded.root --python_filename=embed.py --customise=TauAnalysis/MCEmbeddingTools/embeddingCustomizeAll -n 10
import FWCore.ParameterSet.Config as cms

process = cms.Process('L1ETMSingleMuScanner')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('TrackingTools.TrackAssociator.DetIdAssociatorESProducer_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1))

# Define input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(['/store/user/aburgmei/embedding/20130125-SingleMuProd/SingleMuProd_%d_SingleMu.root' % x for x in range(0,1000)]))
    #fileNames = cms.untracked.vstring(['/store/user/aburgmei/embedding/20130125-SingleMuProd/SingleMuProd_%d_SingleMu.root' % x for x in [41,66]]))
    #fileNames = cms.untracked.vstring(['/store/user/aburgmei/embedding/20130125-SingleMuProd/SingleMuProd_1_SingleMu.root']))
    #fileNames = cms.untracked.vstring('file:/scratch/hh/lustre/cms/user/aburgmei/CMSSW_5_3_2_patch4/src/TauAnalysis/MCEmbeddingTools/test/DYprod/SingleMu.root'))

process.TFileService = cms.Service("TFileService", 
	fileName = cms.string("l1ana.root"))

process.GlobalTag.globaltag = 'START53_V7A::All'

from TrackingTools.TrackAssociator.default_cfi import TrackAssociatorParameterBlock
#from TrackingTools.TrackAssociator.DetIdAssociatorESProducer_cff import *
process.muonCaloDistances = cms.EDProducer('MuonCaloDistanceProducer',
	trackAssociator = TrackAssociatorParameterBlock.TrackAssociatorParameters,
	selectedMuons = cms.InputTag("muons"))

process.anaMuonCalo = cms.EDAnalyzer('AnaL1CaloCleaner',
	caloLengthsPlus = cms.InputTag('muonCaloDistances', 'distancesMuPlus'),
	caloLengthsMinus = cms.InputTag('muonCaloDistances', 'distancesMuMinus'),
	caloDepositsPlus = cms.InputTag('muonCaloDistances', 'depositsMuPlus'),
	caloDepositsMinus = cms.InputTag('muonCaloDistances', 'depositsMuMinus'),
	l1ETM = cms.InputTag('l1extraParticles', 'MET'),
	caloMET = cms.InputTag('metNoHF'),
	genParticles = cms.InputTag('genParticles'),
	muons = cms.InputTag('muons'))

process.p = cms.Path(process.muonCaloDistances*process.anaMuonCalo)

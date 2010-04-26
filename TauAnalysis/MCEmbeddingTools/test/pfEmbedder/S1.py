# -*- coding: utf-8 -*-
import FWCore.ParameterSet.Config as cms

process = cms.Process("EXAMPLE")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )

process.TFileService = cms.Service("TFileService",  fileName = cms.string("histo.root")          )

  
process.load("Configuration.Generator.PythiaUESettings_cfi")

# the following lines are required by hit tracking
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Geometry.CaloEventSetup.CaloTopology_cfi")
process.load("Configuration.StandardSequences.Reconstruction_cff")

process.load("IOMC/RandomEngine/IOMC_cff")
process.RandomNumberGeneratorService.newSource =  cms.PSet(
         initialSeed = cms.untracked.uint32(123456789),
         engineName = cms.untracked.string('HepJamesRandom')
) 

#process.GlobalTag.globaltag = 'MC_31X_V5::All' # 31x
process.GlobalTag.globaltag = 'MC_3XY_V26::All' # 356
#process.GlobalTag.globaltag = 'STARTUP3X_V8A::All' # 33x
#process.GlobalTag.globaltag = 'STARTUP31X_V4::All' # 31x


#TauolaDefaultInputCards = cms.PSet(
##InputCards = cms.vstring('TAUOLA = 0 0 102 ! TAUOLA ')      # 114=l+jet, 102=only muons
#   pjak1 = cms.int32(0),
#   pjak2 = cms.int32(0),
#   mdtau = cms.int32(216)
#)


TauolaNoPolar = cms.PSet(
    UseTauolaPolarization = cms.bool(False)
)
TauolaPolar = cms.PSet(
   UseTauolaPolarization = cms.bool(True)
)



process.load("TauAnalysis.MCEmbeddingTools.MCParticleReplacer_cfi")
process.newSource.algorithm = "ZTauTau"
process.newSource.ZTauTau.TauolaOptions.InputCards.mdtau = cms.int32(216)
process.newSource.ZTauTau.minVisibleTransverseMomentum = cms.untracked.double(0)
#process.newSource.ZTauTau.ExternalDecays.TauolaDefaultInputCards.mdtau = 214

#process.newSource.verbose = True

process.source = cms.Source("PoolSource",
        skipEvents = cms.untracked.uint32(0),
        fileNames = cms.untracked.vstring('file:/tmp/fruboes/OneFileFromEmbSkim/patLayer1_fromAOD_PF2PAT_full_9_2.root')
)

process.load("Configuration.EventContent.EventContent_cff")
process.load("FWCore.MessageService.MessageLogger_cfi")
#process.MessageLogger.cerr.threshold = 'DEBUG'
process.MessageLogger.destinations = cms.untracked.vstring('log')
#process.MessageLogger.log = cms.untracked.PSet( threshold = cms.untracked.string("DEBUG") )
#process.MessageLogger.debugModules = cms.untracked.vstring("*")
    

process.OUTPUT = cms.OutputModule("PoolOutputModule",
# outputCommands = cms.untracked.vstring("drop *", 
#         "keep *_*_*_EXAMPLE",
#         "keep recoTracks_*_*_*",
#         "keep recoMuons_*_*_*"
#         ),
        SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('p1')),
        fileName = cms.untracked.string('file:zMuMuEmbed_output.root')
#         fileName = cms.untracked.string('/tmp/fruboes/Zmumu/zMuMuEmbed_output.root')
)


#process.dump = cms.EDAnalyzer("EventContentAnalyzer")
process.dump = cms.EDAnalyzer("PrintGenMuons")
#  : _etaMax(iConfig.getUntrackedParameter<double>("etaMax")),
#    _ptMin(iConfig.getUntrackedParameter<double>("ptMin")),
#    _pfColl(iConfig.getUntrackedParameter<edm::InputTag>("pfCol"))

process.filter = cms.EDFilter("SelectZmumuevents",
    etaMax = cms.untracked.double(2.2),
    ptMin = cms.untracked.double(10),
    pfCol = cms.untracked.InputTag("particleFlow")
)

#process.raw2digi_step = cms.Path(process.RawToDigi)
#process.reconstruction_step = cms.Path(process.reconstruction)

#process.p1 = cms.Path(process.selectMuons*process.newSource)
#process.p1 = cms.Path(process.dimuonsGlobal*process.selectMuons*process.newSource)

process.filterEmptyEv = cms.EDFilter("EmptyEventsFilter",
    minEvents = cms.untracked.int32(1),
    target =  cms.untracked.int32(1) 
)

process.adaptedMuonsFromDiTauCands = cms.EDProducer("CompositePtrCandidateT1T2MEtAdapter",
    diTau  = cms.untracked.InputTag("zMuMuCandsMuEta"),
    pfCands = cms.untracked.InputTag("particleFlow","")
)

process.dimuonsGlobal = cms.EDProducer('ZmumuPFEmbedder',
    etaMax = cms.untracked.double(2.2),
    ptMin = cms.untracked.double(10),
    tracks = cms.InputTag("generalTracks"),
    selectedMuons = cms.InputTag("adaptedMuonsFromDiTauCands","zMusExtracted")
)

process.generator = process.newSource.clone()
#process.generator.src = cms.InputTag("dimuonsGlobal","zMusExtracted")
process.generator.src = cms.InputTag("adaptedMuonsFromDiTauCands","zMusExtracted")

process.p1 = cms.Path(process.adaptedMuonsFromDiTauCands*process.dimuonsGlobal*process.generator*process.filterEmptyEv)
#process.p1 = cms.Path(process.filter*process.dimuonsGlobal*process.generator)

#process.p1 = cms.Path(process.RawToDigi*process.reconstruction*process.selectMuonHits)

process.outpath = cms.EndPath(process.OUTPUT)
#




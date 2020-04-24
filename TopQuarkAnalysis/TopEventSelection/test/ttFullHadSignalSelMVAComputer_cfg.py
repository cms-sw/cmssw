import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

## add message logger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = 'INFO'
process.MessageLogger.cerr.FwkReport.reportEvery = 1000

## define input
from TopQuarkAnalysis.TopEventProducers.tqafInputFiles_cff import relValTTbar
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(relValTTbar)
)
#process.source = cms.Source("PoolSource",
    #fileNames = cms.untracked.vstring(
    ##'/store/relval/CMSSW_7_1_0_pre1/RelValProdTTbar/GEN-SIM-RECO/START70_V5-v1/00000/14842A6B-2086-E311-B5CB-02163E00E8DA.root'

    ##'/store/user/eschliec/Summer09/7TeV/TTBar/MCatNLO/patTuple_1.root',
    ##'/store/user/eschliec/Summer09/7TeV/TTBar/MCatNLO/patTuple_2.root'

    #'/store/user/eschliec/Summer09/7TeV/QCD/pt0015-pythia/patTuple_1.root',
    #'/store/user/eschliec/Summer09/7TeV/QCD/pt0015-pythia/patTuple_4.root',
    #'/store/user/eschliec/Summer09/7TeV/QCD/pt0015-pythia/patTuple_8.root',
    #'/store/user/eschliec/Summer09/7TeV/QCD/pt0015-pythia/patTuple_7.root',
    #'/store/user/eschliec/Summer09/7TeV/QCD/pt0015-pythia/patTuple_3.root',
    #'/store/user/eschliec/Summer09/7TeV/QCD/pt0015-pythia/patTuple_6.root',
    #'/store/user/eschliec/Summer09/7TeV/QCD/pt0015-pythia/patTuple_9.root',
    #'/store/user/eschliec/Summer09/7TeV/QCD/pt0015-pythia/patTuple_2.root',
    #'/store/user/eschliec/Summer09/7TeV/QCD/pt0015-pythia/patTuple_5.root',
    #'/store/user/eschliec/Summer09/7TeV/QCD/pt0015-pythia/patTuple_10.root',
    #'/store/user/eschliec/Summer09/7TeV/QCD/pt0015-pythia/patTuple_14.root',
    #'/store/user/eschliec/Summer09/7TeV/QCD/pt0015-pythia/patTuple_12.root',
    #'/store/user/eschliec/Summer09/7TeV/QCD/pt0015-pythia/patTuple_13.root',
    #'/store/user/eschliec/Summer09/7TeV/QCD/pt0015-pythia/patTuple_11.root'

    ##'/store/user/eschliec/Summer09/7TeV/QCD/pt1400-pythia/patTuple_1.root',
    ##'/store/user/eschliec/Summer09/7TeV/QCD/pt1400-pythia/patTuple_4.root',
    ##'/store/user/eschliec/Summer09/7TeV/QCD/pt1400-pythia/patTuple_8.root',
    ##'/store/user/eschliec/Summer09/7TeV/QCD/pt1400-pythia/patTuple_7.root',
    ##'/store/user/eschliec/Summer09/7TeV/QCD/pt1400-pythia/patTuple_3.root',
    ##'/store/user/eschliec/Summer09/7TeV/QCD/pt1400-pythia/patTuple_6.root',
    ##'/store/user/eschliec/Summer09/7TeV/QCD/pt1400-pythia/patTuple_9.root',
    ##'/store/user/eschliec/Summer09/7TeV/QCD/pt1400-pythia/patTuple_2.root',
    ##'/store/user/eschliec/Summer09/7TeV/QCD/pt1400-pythia/patTuple_5.root',
    ##'/store/user/eschliec/Summer09/7TeV/QCD/pt1400-pythia/patTuple_10.root',
    ##'/store/user/eschliec/Summer09/7TeV/QCD/pt1400-pythia/patTuple_14.root',
    ##'/store/user/eschliec/Summer09/7TeV/QCD/pt1400-pythia/patTuple_12.root',
    ##'/store/user/eschliec/Summer09/7TeV/QCD/pt1400-pythia/patTuple_13.root',
    ##'/store/user/eschliec/Summer09/7TeV/QCD/pt1400-pythia/patTuple_11.root',
    ##'/store/user/eschliec/Summer09/7TeV/QCD/pt1400-pythia/patTuple_15.root',
    ##'/store/user/eschliec/Summer09/7TeV/QCD/pt1400-pythia/patTuple_16.root',
    ##'/store/user/eschliec/Summer09/7TeV/QCD/pt1400-pythia/patTuple_17.root',
    ##'/store/user/eschliec/Summer09/7TeV/QCD/pt1400-pythia/patTuple_18.root',
    ##'/store/user/eschliec/Summer09/7TeV/QCD/pt1400-pythia/patTuple_19.root',
    ##'/store/user/eschliec/Summer09/7TeV/QCD/pt1400-pythia/patTuple_20.root',
    ##'/store/user/eschliec/Summer09/7TeV/QCD/pt1400-pythia/patTuple_24.root',
    ##'/store/user/eschliec/Summer09/7TeV/QCD/pt1400-pythia/patTuple_22.root',
    ##'/store/user/eschliec/Summer09/7TeV/QCD/pt1400-pythia/patTuple_23.root',
    ##'/store/user/eschliec/Summer09/7TeV/QCD/pt1400-pythia/patTuple_21.root'
     #),
     #skipEvents = cms.untracked.uint32(0)
#)
## define maximal number of events to loop over
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10000)
)
## configure process options
process.options = cms.untracked.PSet(
    wantSummary      = cms.untracked.bool(True)
)

## configure geometry & conditions
process.load("Configuration.Geometry.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc')
process.load("Configuration.StandardSequences.MagneticField_cff")

process.task = cms.Task()

## std sequence for PAT
process.load("PhysicsTools.PatAlgos.producersLayer1.patCandidates_cff")
process.task.add(process.patCandidatesTask)
process.load("PhysicsTools.PatAlgos.selectionLayer1.selectedPatCandidates_cff")
process.task.add(process.selectedPatCandidatesTask)
process.load("TopQuarkAnalysis.TopEventSelection.TtFullHadSignalSelMVAComputer_cff")
process.task.add(process.findTtFullHadSignalSelMVA)

## jet count filter
process.load("PhysicsTools.PatAlgos.selectionLayer1.jetCountFilter_cfi")
process.task.add(process.countPatJets)

## setup jet selection collection
# This configuration is broken because countLayer1Jets is not defined anywhere
process.leadingJetSelection = process.countLayer1Jets.clone(src = 'selectedLayer1Jets',
                                                            minNumber = 6
                                                            )

## path1
process.p = cms.Path(process.leadingJetSelection *
                     process.findTtFullHadSignalSelMVA,
                     process.task
                     )

## output module
process.out = cms.OutputModule(
  "PoolOutputModule",
  SelectEvents = cms.untracked.PSet( SelectEvents = cms.vstring('p')),
  outputCommands = cms.untracked.vstring('drop *',
                                         'keep double_*_DiscSel_*'
                                        ),
  fileName = cms.untracked.string('ttFullHadSignalSelMVAComputer.root')
)
## output path
process.outpath = cms.EndPath(process.out)

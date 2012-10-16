import FWCore.ParameterSet.Config as cms

def customise(process):

  process.simHcalUnsuppressedDigis.doThermalNoise = cms.bool(False)
  #process.simSiStripDigis.CommonModeNoise = cms.bool(False)
  #process.simSiStripDigis.SingleStripNoise = cms.bool(False)
  #process.simSiStripDigis.Noise = cms.bool(False)
  #process.simSiPixelDigis.AddNoise = cms.bool(False)
  #process.simMuonRPCDigis.Noise = cms.bool(False)
  #process.simMuonCSCDigis.strips.doCorrelatedNoise = cms.bool(False)
  #process.simMuonCSCDigis.wires.doNoise = cms.bool(False)
  #process.simMuonCSCDigis.strips.doNoise = cms.bool(False)
  process.simEcalUnsuppressedDigis.    doESNoise = cms.bool(False)
  process.simCastorDigis.    doNoise = cms.bool(False)
  process.simEcalUnsuppressedDigis.    doNoise = cms.bool(False)
  process.simHcalUnsuppressedDigis.    doNoise = cms.bool(False)

  process.TFileService = cms.Service("TFileService",
      fileName = cms.string("histo.root"),
      closeFileFast = cms.untracked.bool(True)
  )

  from TrackingTools.TrackAssociator.default_cfi import TrackAssociatorParameterBlock

  myBlock=TrackAssociatorParameterBlock.clone()
  myBlock.TrackAssociatorParameters.usePreshower = cms.bool(True)

  process.ana =  cms.EDProducer('MuonCaloCleaner',
     myBlock,
     selectedMuons = cms.InputTag("muons"),
     storeDeps = cms.untracked.bool(True)
  )


  process.anaPlus = cms.EDAnalyzer("AnaMuonCaloCleaner",
    colLen = cms.InputTag("ana", "plus"),
    colDep = cms.InputTag("ana", "plusDeposits"),
    selectedMuons = cms.InputTag("muons"),
    charge = cms.int32(1)
  )

  process.anaMinus = cms.EDAnalyzer("AnaMuonCaloCleaner",
    colLen = cms.InputTag("ana", "minus"),
    colDep = cms.InputTag("ana", "minusDeposits"),
    selectedMuons = cms.InputTag("muons"),
    charge = cms.int32(-1)
  )  

  process.mySeq = cms.Sequence(process.ProductionFilterSequence*process.ana*process.anaPlus*process.anaMinus)
  process.myPath = cms.Path(process.mySeq)

  process.schedule.extend([process.myPath])

  process.MessageLogger.cerr.FwkReport.reportEvery = 100

  return process

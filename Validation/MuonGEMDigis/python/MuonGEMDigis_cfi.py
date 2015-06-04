import FWCore.ParameterSet.Config as cms

gemStripValidation = cms.EDAnalyzer('GEMStripDigiValidation',
  outputFile = cms.string(''),
  stripLabel= cms.InputTag('simMuonGEMDigis'),
  simInputLabel = cms.InputTag('g4SimHits',"MuonGEMHits"),
  nBinGlobalZR = cms.untracked.vdouble(200,200,200,150,180,250), 
  RangeGlobalZR = cms.untracked.vdouble(564,572,786,794,794,802,110,260,170,350,100,350), 
  nBinGlobalXY = cms.untracked.int32(360),
  detailPlot = cms.bool(False), 
)
gemPadValidation = cms.EDAnalyzer('GEMPadDigiValidation',
  outputFile = cms.string(''),
  PadLabel = cms.InputTag('simMuonGEMPadDigis'),
  simInputLabel = cms.InputTag('g4SimHits',"MuonGEMHits"),
  nBinGlobalZR = cms.untracked.vdouble(200,200,200,150,180,250), 
  RangeGlobalZR = cms.untracked.vdouble(564,572,786,794,794,802,110,260,170,350,100,350), 
  nBinGlobalXY = cms.untracked.int32(360), 
  detailPlot = cms.bool(False), 
)
gemCoPadValidation = cms.EDAnalyzer('GEMCoPadDigiValidation',
  outputFile = cms.string(''),
  CopadLabel = cms.InputTag('simCscTriggerPrimitiveDigis') ,
  simInputLabel = cms.InputTag('g4SimHits',"MuonGEMHits"),
  nBinGlobalZR = cms.untracked.vdouble(200,200,200,150,180,250), 
  RangeGlobalZR = cms.untracked.vdouble(564,572,786,794,794,802,110,260,170,350,100,350), 
  nBinGlobalXY = cms.untracked.int32(360), 
  detailPlot = cms.bool(False), 
)

gemDigiTrackValidation = cms.EDAnalyzer('GEMDigiTrackMatch',
  simInputLabel = cms.untracked.string('g4SimHits'),
  simTrackCollection = cms.InputTag('g4SimHits'),
  simVertexCollection = cms.InputTag('g4SimHits'),
  verboseSimHit = cms.untracked.int32(0),
  # GEM digi matching:
  verboseGEMDigi = cms.untracked.int32(0),
  gemDigiInput = cms.InputTag("simMuonGEMDigis"),
  gemPadDigiInput = cms.InputTag("simMuonGEMPadDigis"),
  gemCoPadDigiInput = cms.InputTag("simCscTriggerPrimitiveDigis"),
  minBXGEM = cms.untracked.int32(-1),
  maxBXGEM = cms.untracked.int32(1),
  matchDeltaStripGEM = cms.untracked.int32(1),
  gemMinPt = cms.untracked.double(5.0),
  gemMinEta = cms.untracked.double(1.55),
  gemMaxEta = cms.untracked.double(2.45),
)

gemGeometryChecker = cms.EDAnalyzer('GEMCheckGeometry')

gemDigiValidation = cms.Sequence( gemStripValidation+gemPadValidation+gemCoPadValidation+gemDigiTrackValidation+gemGeometryChecker)

me11tmbSLHCGEM = cms.PSet(
        mpcBlockMe1a    = cms.uint32(0),
        alctTrigEnable  = cms.uint32(0),
        clctTrigEnable  = cms.uint32(0),
        matchTrigEnable = cms.uint32(1),
        matchTrigWindowSize = cms.uint32(3),
        tmbL1aWindowSize = cms.uint32(7),
        verbosity = cms.int32(0),
        tmbEarlyTbins = cms.int32(4),
        tmbReadoutEarliest2 = cms.bool(False),
        tmbDropUsedAlcts = cms.bool(False),
        clctToAlct = cms.bool(False),
        tmbDropUsedClcts = cms.bool(False),
        matchEarliestAlctME11Only = cms.bool(False),
        matchEarliestClctME11Only = cms.bool(False),
        tmbCrossBxAlgorithm = cms.uint32(2),
        maxME11LCTs = cms.uint32(2),

        ## run in debug mode
        debugLUTs = cms.bool(False),
        debugMatching = cms.bool(False),
        debugGEMDphi = cms.bool(False),

        ## use old dataformat
        useOldLCTDataFormat = cms.bool(True),

        ## copad construction
        maxDeltaBXInCoPad = cms.int32(1),
        maxDeltaPadInCoPad = cms.int32(1),

        ## matching to pads in case LowQ CLCT
        maxDeltaBXPadEven = cms.int32(1),
        maxDeltaBXPadOdd = cms.int32(1),
        maxDeltaPadPadEven = cms.int32(2),
        maxDeltaPadPadOdd = cms.int32(3),

        ## matching to pads in case absent CLCT
        maxDeltaBXCoPadEven = cms.int32(0),
        maxDeltaBXCoPadOdd = cms.int32(0),
        maxDeltaPadCoPadEven = cms.int32(2),
        maxDeltaPadCoPadOdd = cms.int32(3),

        ## efficiency recovery switches
        dropLowQualityCLCTsNoGEMs_ME1a = cms.bool(False),
        dropLowQualityCLCTsNoGEMs_ME1b = cms.bool(True),
        dropLowQualityALCTsNoGEMs_ME1a = cms.bool(False),
        dropLowQualityALCTsNoGEMs_ME1b = cms.bool(False),
        buildLCTfromALCTandGEM_ME1a = cms.bool(True),
        buildLCTfromALCTandGEM_ME1b = cms.bool(True),
        buildLCTfromCLCTandGEM_ME1a = cms.bool(False),
        buildLCTfromCLCTandGEM_ME1b = cms.bool(False),
        doLCTGhostBustingWithGEMs = cms.bool(False),
        correctLCTtimingWithGEM = cms.bool(False),
        promoteALCTGEMpattern = cms.bool(True),
        promoteALCTGEMquality = cms.bool(True),
        promoteCLCTGEMquality_ME1a = cms.bool(True),
        promoteCLCTGEMquality_ME1b = cms.bool(True),
        
        ## rate reduction 
        doGemMatching = cms.bool(True),
        gemMatchDeltaEta = cms.double(0.08),
        gemMatchDeltaBX = cms.int32(1),
        gemMatchDeltaPhiOdd = cms.double(1),
        gemMatchDeltaPhiEven = cms.double(1),
        gemMatchMinEta = cms.double(1.55),
        gemMatchMaxEta = cms.double(2.15),
        gemClearNomatchLCTs = cms.bool(False),

        ## cross BX algorithm
        firstTwoLCTsInChamber = cms.bool(True),
)

me21tmbSLHCGEM = cms.PSet(
        mpcBlockMe1a    = cms.uint32(0),
        alctTrigEnable  = cms.uint32(0),
        clctTrigEnable  = cms.uint32(0),
        matchTrigEnable = cms.uint32(1),
        matchTrigWindowSize = cms.uint32(3),
        tmbL1aWindowSize = cms.uint32(7),
        verbosity = cms.int32(0),
        tmbEarlyTbins = cms.int32(4),
        tmbReadoutEarliest2 = cms.bool(False),
        tmbDropUsedAlcts = cms.bool(False),
        clctToAlct = cms.bool(False),
        tmbDropUsedClcts = cms.bool(False),
        matchEarliestAlctME21Only = cms.bool(False),
        matchEarliestClctME21Only = cms.bool(False),
        tmbCrossBxAlgorithm = cms.uint32(2),
        maxME21LCTs = cms.uint32(2),

        ## run in debug mode
        debugLUTs = cms.bool(False),
        debugMatching = cms.bool(False),
        debugGEMDphi = cms.bool(False),

        ## use old dataformat
        useOldLCTDataFormat = cms.bool(True),

        ## copad construction
        maxDeltaBXInCoPad = cms.int32(1),
        maxDeltaPadInCoPad = cms.int32(2),

        ## matching to pads in case LowQ CLCT
        maxDeltaBXPad = cms.int32(1),
        maxDeltaPadPadOdd = cms.int32(4),
        maxDeltaPadPadEven = cms.int32(3),
        maxDeltaWg = cms.int32(2),

        ## matching to pads in case absent CLCT
        maxDeltaBXCoPad = cms.int32(1),
        maxDeltaPadCoPad = cms.int32(2),

        ## efficiency recovery switches
        dropLowQualityALCTsNoGEMs = cms.bool(False),
        dropLowQualityCLCTsNoGEMs = cms.bool(True),
        buildLCTfromALCTandGEM = cms.bool(True),
        buildLCTfromCLCTandGEM = cms.bool(False),
        doLCTGhostBustingWithGEMs = cms.bool(False),
        correctLCTtimingWithGEM = cms.bool(False),
        promoteALCTGEMpattern = cms.bool(True),
        promoteALCTGEMquality = cms.bool(True),
        promoteCLCTGEMquality = cms.bool(True),

        ## rate reduction 
        doGemMatching = cms.bool(True),
        gemMatchDeltaEta = cms.double(0.08),
        gemMatchDeltaBX = cms.int32(1),
        gemMatchDeltaPhiOdd = cms.double(1),
        gemMatchDeltaPhiEven = cms.double(1),
        gemMatchMinEta = cms.double(1.5),
        gemMatchMaxEta = cms.double(2.45),
        gemClearNomatchLCTs = cms.bool(False),

        firstTwoLCTsInChamber = cms.bool(True),
    )

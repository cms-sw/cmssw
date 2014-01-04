import FWCore.ParameterSet.Config as cms

from GEMCode.GEMValidation.simTrackMatching_cfi import SimTrackMatching

GEMCSCTriggerEfficiency = cms.EDAnalyzer("GEMCSCTriggerEfficiency",
    doME1a = cms.untracked.bool(True),
    defaultME1a = cms.untracked.bool(False),
    gangedME1a = cms.untracked.bool(False),
    ## BX windows - about central BX=6                                     
    minBxALCT = cms.untracked.int32(5),
    maxBxALCT = cms.untracked.int32(7),
    minBxCLCT = cms.untracked.int32(5),
    maxBxCLCT = cms.untracked.int32(7),
    minBxLCT = cms.untracked.int32(5),
    maxBxLCT = cms.untracked.int32(7),
    minBxMPLCT = cms.untracked.int32(5),
    maxBxMPLCT = cms.untracked.int32(7),
    ## matching                                     
    doStrictSimHitToTrackMatch = cms.untracked.bool(True),
    matchAllTrigPrimitivesInChamber = cms.untracked.bool(True),
    minDeltaYAnode = cms.untracked.double(-1),
    minDeltaYCathode = cms.untracked.double(-1),
    minDeltaWire = cms.untracked.int32(-2),
    maxDeltaWire = cms.untracked.int32(2),
    minDeltaStrip = cms.untracked.int32(2),
    lightRun = cms.untracked.bool(False),
    ## looser requirement on the number of chamber hits
    minNHitsChamber = cms.untracked.int32(4),
    minNStWithMinHitsinChamber = cms.untracked.int32(0),
    requireME1With4Hits = cms.untracked.bool(False),
    ## bending angles                                     
    gemPTs = cms.vdouble(0., 5., 6., 10., 15., 20., 30., 40.),
    gemDPhisOdd = cms.vdouble(1., 0.02203511, 0.0182579,   0.01066 , 0.00722795 , 0.00562598 , 0.00416544 , 0.00342827),
    gemDPhisEven = cms.vdouble(1., 0.00930056, 0.00790009, 0.00483286, 0.0036323, 0.00304879, 0.00253782, 0.00230833),
    ## simtrack cuts
    minSimTrPt = cms.untracked.double(2.),
    minSimTrPhi = cms.untracked.double(-3.15),
    maxSimTrPhi = cms.untracked.double(3.15),
    minSimTrEta = cms.untracked.double(1.45),
    maxSimTrEta = cms.untracked.double(2.5),
    minSimTrackDR = cms.untracked.double(0.0),
    invertSimTrPhiEta = cms.untracked.bool(False),
    onlyForwardMuons = cms.untracked.bool(True),
    goodChambersOnly = cms.untracked.bool(False),
    sectorProcessor = cms.untracked.PSet(),
    simTrackMatching = SimTrackMatching,
    strips = cms.untracked.PSet(),
    ## debuggin purposes                                     
    debugALLEVENT = cms.untracked.int32(0),
    debugINHISTOS = cms.untracked.int32(0),
    debugALCT     = cms.untracked.int32(0),
    debugCLCT     = cms.untracked.int32(0),
    debugLCT      = cms.untracked.int32(0),
    debugMPLCT    = cms.untracked.int32(0),
    debugTFTRACK  = cms.untracked.int32(0),
    debugTFCAND   = cms.untracked.int32(0),
    debugGMTCAND  = cms.untracked.int32(0),
    debugL1EXTRA  = cms.untracked.int32(0),

    ## collecting the params per collection
    ## not used at the moment
    simTrack = cms.untracked.PSet(
        minPt = cms.untracked.double(2.),
        minPhi = cms.untracked.double(-3.15),
        maxPhi = cms.untracked.double(3.15),
        minEta = cms.untracked.double(1.45),
        maxEta = cms.untracked.double(2.5),
        minDR = cms.untracked.double(0.0),
        invertPhiEta = cms.untracked.bool(False),
        onlyForwardMuons = cms.untracked.bool(True),
        doStrictSimHitToTrackMatch = cms.untracked.bool(True),
    ),
    cscSimhits = cms.untracked.PSet(
        debug = cms.untracked.bool(False),
    ),
    gemSimhits = cms.untracked.PSet(
        debug = cms.untracked.bool(False),
    ),
    wireDigis = cms.untracked.PSet(
        debug = cms.untracked.bool(False),
        minDelta = cms.untracked.int32(-2),
        maxDelta = cms.untracked.int32(2),
        gangedME1a = cms.untracked.bool(False),
    ),
    stripDigis = cms.untracked.PSet(
        debug = cms.untracked.bool(False),
        ## find out why minDeltaStrip positive is!!
        minDelta = cms.untracked.int32(2),
        maxDelta = cms.untracked.int32(2),
    ),
    alcts = cms.untracked.PSet(
        debug = cms.untracked.bool(False),
        minBX = cms.untracked.int32(5),
        maxBX = cms.untracked.int32(7),
    ),                                    
    clcts = cms.untracked.PSet(
        debug = cms.untracked.bool(False),
        minBX = cms.untracked.int32(5),
        maxBX = cms.untracked.int32(7),
    ),                                    
    lcts = cms.untracked.PSet(
        debug = cms.untracked.bool(False),
        minBX = cms.untracked.int32(5),
        maxBX = cms.untracked.int32(7),
        addGhosts = cms.untracked.bool(True)
    ),
    mplcts = cms.untracked.PSet(
        debug = cms.untracked.bool(False),
        minBX = cms.untracked.int32(5),
        maxBX = cms.untracked.int32(7),
    ),
    tfTrack = cms.untracked.PSet(
        debug = cms.untracked.bool(False),
    ),
    tfCand = cms.untracked.PSet(
        debug = cms.untracked.bool(False),
    ),
    gmtCand = cms.untracked.PSet(
        debug = cms.untracked.bool(False),
        minBX = cms.untracked.int32(-1),
        maxBX = cms.untracked.int32(1),
    ),
    l1Extra = cms.untracked.PSet(
        debug = cms.untracked.bool(False),
    ),                                    
)

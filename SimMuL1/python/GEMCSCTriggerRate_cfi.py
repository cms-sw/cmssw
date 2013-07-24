import FWCore.ParameterSet.Config as cms

GEMCSCTriggerRate = cms.EDFilter("GEMCSCTriggerRate",
    doStrictSimHitToTrackMatch = cms.untracked.bool(True),
    matchAllTrigPrimitivesInChamber = cms.untracked.bool(True),
    
    minDeltaWire = cms.untracked.int32(-2),
    maxDeltaWire = cms.untracked.int32(2),
    minDeltaStrip = cms.untracked.int32(2),
    
    minNStWith4Hits = cms.untracked.int32(0),
    requireME1With4Hits = cms.untracked.bool(False),
    
    ## add option to include the bending angle library in here!!!                                     
    gemPTs = cms.vdouble(0., 6., 10., 15., 20., 30., 40.),
    gemDPhisOdd = cms.vdouble(1., 0.0182579,   0.01066 , 0.00722795 , 0.00562598 , 0.00416544 , 0.00342827),
    gemDPhisEven = cms.vdouble(1., 0.00790009, 0.00483286, 0.0036323, 0.00304879, 0.00253782, 0.00230833),

    doME1a = cms.untracked.bool(True),
    defaultME1a = cms.untracked.bool(False),
    gangedME1a = cms.untracked.bool(False),
    
    minBxALCT = cms.untracked.int32(5),
    maxBxALCT = cms.untracked.int32(7),
    minBxCLCT = cms.untracked.int32(5),
    maxBxCLCT = cms.untracked.int32(7),
    minBxLCT = cms.untracked.int32(5),
    maxBxLCT = cms.untracked.int32(7),
    minBxMPLCT = cms.untracked.int32(5),
    maxBxMPLCT = cms.untracked.int32(7),

    minSimTrDR = cms.untracked.double(0.0),
    minSimTrPt = cms.untracked.double(2.),
    minSimTrPhi = cms.untracked.double(-9.),
    maxSimTrPhi = cms.untracked.double(9.),
    minSimTrEta = cms.untracked.double(-2.5),
    maxSimTrEta = cms.untracked.double(2.5),
    invertSimTrPhiEta = cms.untracked.bool(False),
    
    goodChambersOnly = cms.untracked.bool(False),

    simTrackGEMMatching = cms.untracked.PSet(),
    sectorProcessor = cms.untracked.PSet(),
    strips = cms.untracked.PSet()
)

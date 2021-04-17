import FWCore.ParameterSet.Config as cms

CaloParticleSelectionForEfficiency = cms.PSet(
    ptMinCP = cms.double(0.5),
    ptMaxCP = cms.double(300.),
    minRapidityCP = cms.double(-3.1),
    maxRapidityCP = cms.double(3.1),
    #--z position of the origin vertex less than lipCP
    lipCP = cms.double(30.0),
    #-- transverse component squared sum less that tipCP*tipCP
    tipCP = cms.double(60),
    chargedOnlyCP = cms.bool(False),
    stableOnlyCP = cms.bool(False),
    notConvertedOnlyCP = cms.bool(True),
    #311: K0, 130: K0_short, 310: K0_long
    pdgIdCP = cms.vint32(11, -11, 13, -13, 22, 111, 211, -211, 321, -321, 311, 130, 310),
    #--signal only means no PU particles
    signalOnlyCP = cms.bool(True),
    #--intime only means no OOT PU particles
    intimeOnlyCP = cms.bool(True),
    #The total number of rechits
    minHitCP = cms.int32(0),
    maxSimClustersCP = cms.int32(-1)
)

import FWCore.ParameterSet.Config as cms

CaloParticleSelectionForEfficiency = cms.PSet(
    ptMinCP = cms.double(0.005),
    ptMaxCP = cms.double(250.),
    minRapidityCP = cms.double(-4.5),
    maxRapidityCP = cms.double(4.5),
    #--z position of the origin vertex less than lipCP
    lipCP = cms.double(30.0),
    #-- transverse component squared sum less that tipCP*tipCP
    tipCP = cms.double(60),
    chargedOnlyCP = cms.bool(False),
    stableOnlyCP = cms.bool(False),
    pdgIdCP = cms.vint32(11, -11, 13, -13, 22, 111, 211, -211, 321, -321),
    #--signal only means no PU particles
    signalOnlyCP = cms.bool(True),
    #--intime only means no OOT PU particles
    intimeOnlyCP = cms.bool(True),
    #The total number of rechits
    minHitCP = cms.int32(0)
)

  

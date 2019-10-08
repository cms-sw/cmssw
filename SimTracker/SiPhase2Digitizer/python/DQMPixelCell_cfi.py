import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer

# PSet for the histos
ClusterSize1D = cms.PSet(
    Nxbins = cms.int32(201),
    xmin = cms.double(-0.5),
    xmax = cms.double(199.5)
    )
Charge1D = cms.PSet(
    Nxbins = cms.int32(201),
    xmin = cms.double(-0.5),
    xmax = cms.double(199.5)
    )
TrackDxdz = cms.PSet(
    Nxbins = cms.int32(100),
    xmin = cms.double(-0.5),
    xmax = cms.double(0.5)
    )
TrackDydz = cms.PSet(
    Nxbins = cms.int32(101),
    xmin = cms.double(-0.5),
    xmax = cms.double(0.5)
    )
Position = cms.PSet(
    Nxbins = cms.int32(300),
    Nybins = cms.int32(300)
    )
Efficiency = cms.PSet(
    Nxbins = cms.int32(400),
    Nybins = cms.int32(100),
    zmin   = cms.double(0),
    zmax   = cms.double(1)
    )
ClusterSize = cms.PSet(
    Nxbins = cms.int32(400),
    Nybins = cms.int32(100),
    zmin   = cms.double(0),
    zmax   = cms.double(10)
    )
Charge = cms.PSet(
    Nxbins = cms.int32(400),
    Nybins = cms.int32(100),
    zmin   = cms.double(0),
    zmax   = cms.double(20)
    )

dqmcell = DQMEDAnalyzer('DQMPixelCell',
    TopFolderName = cms.string("PixelCell"),
    PixelDigiSource = cms.InputTag("simSiPixelDigis","Pixel"),
    PixelDigiSimSource = cms.InputTag("simSiPixelDigis", "Pixel"),
    PSimHitSource  = cms.VInputTag('g4SimHits:TrackerHitsPixelBarrelLowTof',
                                   'g4SimHits:TrackerHitsPixelBarrelHighTof',
                                   'g4SimHits:TrackerHitsPixelEndcapLowTof',
                                   'g4SimHits:TrackerHitsPixelEndcapHighTof',
                                   'g4SimHits:TrackerHitsTIBLowTof',
                                   'g4SimHits:TrackerHitsTIBHighTof',
                                   'g4SimHits:TrackerHitsTIDLowTof',
                                   'g4SimHits:TrackerHitsTIDHighTof',
                                   'g4SimHits:TrackerHitsTOBLowTof',
                                   'g4SimHits:TrackerHitsTOBHighTof',
                                   'g4SimHits:TrackerHitsTECLowTof',
                                   'g4SimHits:TrackerHitsTECHighTof'),
    SimTrackSource = cms.InputTag("g4SimHits"),
    GeometryType = cms.string('idealForDigi'),

    ClusterSize1D = ClusterSize1D.clone(),
    Charge1D = Charge1D.clone(),

    Position_0 = Position.clone(),
    Position_1 = Position.clone(),
    Position_2 = Position.clone(),
    
    Efficiency_0 = Efficiency.clone(),
    Efficiency_1 = Efficiency.clone(),
    Efficiency_2 = Efficiency.clone(),
    
    ClusterSize_0 = ClusterSize.clone(),
    ClusterSize_1 = ClusterSize.clone(),
    ClusterSize_2 = ClusterSize.clone(),

    Charge_0 = Charge.clone(),
    Charge_1 = Charge.clone(),
    Charge_2 = Charge.clone(),
    )


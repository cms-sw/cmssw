import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer

Nx=40
Ny=40

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
    Nxbins = cms.int32(1000),
    xmin = cms.double(-3.5),
    xmax = cms.double(3.5)
    )
TrackDydz = cms.PSet(
    Nxbins = cms.int32(1000),
    xmin = cms.double(-3.5),
    xmax = cms.double(3.5)
    )
TrackAngleDxdz = cms.PSet(
    Nxbins = cms.int32(300),
    xmin = cms.double(-1.58),
    xmax = cms.double(1.58)
    )
TrackAngleDydz = cms.PSet(
    Nxbins = cms.int32(300),
    xmin = cms.double(-1.58),
    xmax = cms.double(1.58)
    )
Dx1D = cms.PSet(
    Nxbins = cms.int32(300),
    xmin = cms.double(-50),
    xmax = cms.double(50)
    )
Dy1D = cms.PSet(
    Nxbins = cms.int32(300),
    xmin = cms.double(-150),
    xmax = cms.double(150)
    )
TrackXY = cms.PSet(
    Nxbins = cms.int32(1250),
    Nybins = cms.int32(1250),
    xmin   = cms.double(-12500),
    xmax   = cms.double(12500),
    ymin   = cms.double(-12500),
    ymax   = cms.double(12500)
    )
TrackRZ = cms.PSet(
    Nxbins = cms.int32(3000),
    Nybins = cms.int32(1250),
    xmin   = cms.double(-30000),
    xmax   = cms.double(30000),
    ymin   = cms.double(0),
    ymax   = cms.double(12500)
    )
Position = cms.PSet(
    Nxbins = cms.int32(300),
    Nybins = cms.int32(300)
    )
Efficiency = cms.PSet(
    Nxbins = cms.int32(Nx),
    Nybins = cms.int32(Ny),
    zmin   = cms.double(0.9),
    zmax   = cms.double(1)
    )
ClusterSize = cms.PSet(
    Nxbins = cms.int32(Nx),
    Nybins = cms.int32(Ny),
    zmin   = cms.double(0),
    zmax   = cms.double(10)
    )
Charge = cms.PSet(
    Nxbins = cms.int32(Nx),
    Nybins = cms.int32(Ny),
    zmin   = cms.double(0),
    zmax   = cms.double(20)
    )
Dx = cms.PSet(
    Nxbins = cms.int32(Nx),
    Nybins = cms.int32(Ny),
    zmin   = cms.double(-50),
    zmax   = cms.double(50)
    )
Dy = cms.PSet(
    Nxbins = cms.int32(Nx),
    Nybins = cms.int32(Ny),
    zmin   = cms.double(-150),
    zmax   = cms.double(150)
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
    TrackDxdz = TrackDxdz.clone(),
    TrackDydz = TrackDydz.clone(),
    TrackAngleDxdz = TrackAngleDxdz.clone(),
    TrackAngleDydz = TrackAngleDydz.clone(),
    TrackXY = TrackXY.clone(),
    TrackRZ = TrackRZ.clone(),
    Dx1D = Dx1D.clone(),
    Dy1D = Dy1D.clone(),

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
    
    Dx_0 = Dx.clone(),
    Dx_1 = Dx.clone(),
    Dx_2 = Dx.clone(),
    
    Dy_0 = Dy.clone(),
    Dy_1 = Dy.clone(),
    Dy_2 = Dy.clone(),
    )


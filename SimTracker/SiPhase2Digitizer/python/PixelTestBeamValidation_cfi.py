import FWCore.ParameterSet.Config as cms
import numpy as np

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer


Nx=60
Ny=60

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
ChargeElec1D = cms.PSet(
    Nxbins = cms.int32(300),
    xmin = cms.double(0),
    xmax = cms.double(120.0e3)
    )
DigiCharge1D = cms.PSet(
    Nxbins = cms.int32(17),
    xmin = cms.double(-0.5),
    xmax = cms.double(16.5)
    )
TrackAngleDxdz = cms.PSet(
    Nxbins = cms.int32(300),
    xmin = cms.double(-np.radians(3.0)),
    xmax = cms.double(np.radians(3.0))
    )
TrackAngleDydz = cms.PSet(
    Nxbins = cms.int32(300),
    xmin = cms.double(-np.radians(3.0)),
    xmax = cms.double(np.radians(3.0))
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
Dxy2D = cms.PSet(
    Nxbins = cms.int32(Nx),
    Nybins = cms.int32(Ny),
    xmin   = cms.double(-50),
    xmax   = cms.double(50),
    ymin   = cms.double(-150),
    ymax   = cms.double(150)
    )
SimClusterCharge = cms.PSet(
    Nxbins = cms.int32(300),
    Nybins = cms.int32(201),
    xmin   = cms.double(0.),
    xmax   = cms.double(300.),
    ymin   = cms.double(0),
    ymax   = cms.double(120.5e3)
    )
TrackXY = cms.PSet(
    Nxbins = cms.int32(1250),
    Nybins = cms.int32(1250),
    xmin   = cms.double(-26.),
    xmax   = cms.double(26.),
    ymin   = cms.double(-26.),
    ymax   = cms.double(26.)
    )
TrackRZ = cms.PSet(
    Nxbins = cms.int32(3000),
    Nybins = cms.int32(1250),
    xmin   = cms.double(-300),
    xmax   = cms.double(300),
    ymin   = cms.double(0),
    ymax   = cms.double(26.)
    )
Position = cms.PSet(
    Nxbins = cms.int32(Nx),
    Nybins = cms.int32(Ny)
    )
Efficiency = cms.PSet(
    Nxbins = cms.int32(Nx),
    Nybins = cms.int32(Ny),
    zmin   = cms.double(0),
    zmax   = cms.double(0)
    )
ClusterSize = cms.PSet(
    Nxbins = cms.int32(Nx),
    Nybins = cms.int32(Ny),
    zmin   = cms.double(0),
    zmax   = cms.double(0)
    )
Charge = cms.PSet(
    Nxbins = cms.int32(Nx),
    Nybins = cms.int32(Ny),
    zmin   = cms.double(0),
    zmax   = cms.double(0)
    )
Dx = cms.PSet(
    Nxbins = cms.int32(Nx),
    Nybins = cms.int32(Ny),
    zmin   = cms.double(0),
    zmax   = cms.double(0)
    )
Dy = cms.PSet(
    Nxbins = cms.int32(Nx),
    Nybins = cms.int32(Ny),
    zmin   = cms.double(0),
    zmax   = cms.double(0)
    )

dqmcell = DQMEDAnalyzer('PixelTestBeamValidation',
    # WARNING: be sure it is the same value used with the Pixel3DDigitizer
    ThresholdInElectrons = cms.double(1000.0),
    ElectronsPerADC   = cms.double(1600.0),
    TracksEntryAngleX = cms.untracked.vdouble(-np.radians(2.0),np.radians(2.0)),
    TracksEntryAngleY = cms.untracked.vdouble(-np.radians(2.0),np.radians(2.0)),
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
    ClusterSize1Dx = ClusterSize1D.clone(),
    ClusterSize1Dy = ClusterSize1D.clone(),
    Charge1D = Charge1D.clone(),
    ChargeElec1D = ChargeElec1D.clone(),
    TrackAngleDxdz = TrackAngleDxdz.clone(),
    TrackAngleDydz = TrackAngleDydz.clone(),
    TrackXY = TrackXY.clone(),
    TrackRZ = TrackRZ.clone(),
    DigiXY = TrackXY.clone(),
    DigiRZ = TrackRZ.clone(),
    Dx1D = Dx1D.clone(),
    Dy1D = Dy1D.clone(),
    Dxy2D = Dxy2D.clone(),
    DigiCharge1D = DigiCharge1D.clone(),
    SimClusterCharge = SimClusterCharge.clone(),

    Position_0 = Position.clone(),
    Position_1 = Position.clone(),
    Position_2 = Position.clone(),
    MatchedPosition_0 = Position.clone(),
    MatchedPosition_1 = Position.clone(),
    MatchedPosition_2 = Position.clone(),
    
    
    Efficiency_0 = Efficiency.clone(),
    Efficiency_1 = Efficiency.clone(),
    Efficiency_2 = Efficiency.clone(),
    
    ClusterSize_0 = ClusterSize.clone(),
    ClusterSize_1 = ClusterSize.clone(),
    ClusterSize_2 = ClusterSize.clone(),

    Charge_0 = Charge.clone(),
    Charge_1 = Charge.clone(),
    Charge_2 = Charge.clone(),
    
    Charge_elec_0 = Charge.clone(),
    Charge_elec_1 = Charge.clone(),
    Charge_elec_2 = Charge.clone(),
    
    Dx_0 = Dx.clone(),
    Dx_1 = Dx.clone(),
    Dx_2 = Dx.clone(),
    
    Dy_0 = Dy.clone(),
    Dy_1 = Dy.clone(),
    Dy_2 = Dy.clone(),
    )


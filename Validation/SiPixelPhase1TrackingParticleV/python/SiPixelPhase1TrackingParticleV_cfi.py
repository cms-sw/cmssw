import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester
from DQM.SiPixelPhase1Common.HistogramManager_cfi import *

SiPixelPhase1TrackingParticleMass = DefaultHisto.clone(
  name = "mass",
  title = "Tracking Particle Mass",
  range_min = -1.0, range_max = 5.0, range_nbins = 100,
  xlabel = "Mass",
  dimensions = 1,
  topFolderName = "PixelPhase1V/TrackingParticle",
  specs = VPSet(
   Specification().groupBy("").save(),
   Specification().groupBy("PXBarrel").save(),
   Specification().groupBy("PXForward").save(),
  )
)

SiPixelPhase1TrackingParticleCharge = SiPixelPhase1TrackingParticleMass.clone(
  name = "charge",
  title = "Tracking Particle Charge",
  range_min = -5, range_max = 5.0, range_nbins = 10,
  xlabel = "Charge",
)

SiPixelPhase1TrackingParticleId = SiPixelPhase1TrackingParticleMass.clone(
  name = "id",
  title = "Tracking Particle Id",
  range_min = -5000, range_max = 5000, range_nbins = 500,
  xlabel = "PID",
)

SiPixelPhase1TrackingParticleNhits = SiPixelPhase1TrackingParticleMass.clone(
  name = "charge",
  title = "Tracking Particle All Hits",
  range_min = -0.5, range_max = 199.5, range_nbins = 200,
  xlabel = "Total # Hits",
)

SiPixelPhase1TrackingParticleMatched = SiPixelPhase1TrackingParticleMass.clone(
  name = "matched",
  title = "Tracking Particle Matched Hits",
  range_min = -0.5, range_max = 99.5, range_nbins = 100,
  xlabel = "Matched Hits",
)

SiPixelPhase1TrackingParticlePt = SiPixelPhase1TrackingParticleMass.clone(
  name = "charge",
  title = "Tracking Particle Pt",
  range_min = 0, range_max = 100, range_nbins = 100,
  xlabel = "Pt",
)

SiPixelPhase1TrackingParticlePhi = SiPixelPhase1TrackingParticleMass.clone(
  name = "phi",
  title = "Tracking Particle Phi",
  range_min = -4, range_max = 4, range_nbins = 100,
  xlabel = "Phi",
)

SiPixelPhase1TrackingParticleEta = SiPixelPhase1TrackingParticleMass.clone(
  name = "eta",
  title = "Tracking Particle Eta",
  range_min = -7, range_max = 7, range_nbins = 100,
  xlabel = "Eta",
)

SiPixelPhase1TrackingParticleVtx = SiPixelPhase1TrackingParticleMass.clone(
  name = "Vtx",
  title = "Tracking Particle VtxX",
  range_min = -100, range_max = 100, range_nbins = 100,
  xlabel = "VtxX",
)

SiPixelPhase1TrackingParticleVty = SiPixelPhase1TrackingParticleMass.clone(
  name = "Vty",
  title = "Tracking Particle VtxY",
  range_min = -100, range_max = 100, range_nbins = 100,
  xlabel = "VtxY",
)

SiPixelPhase1TrackingParticleVtz = SiPixelPhase1TrackingParticleMass.clone(
  name = "Vtz",
  title = "Tracking Particle VtxZ",
  range_min = -100, range_max = 100, range_nbins = 100,
  xlabel = "VtxZ",
)

SiPixelPhase1TrackingParticleTip = SiPixelPhase1TrackingParticleMass.clone(
  name = "tip",
  title = "Tracking Particle tip",
  range_min = 0, range_max = 1000, range_nbins = 100,
  xlabel = "tip",
)

SiPixelPhase1TrackingParticleLip = SiPixelPhase1TrackingParticleMass.clone(
  name = "lip",
  title = "Tracking Particle lip",
  range_min = 0, range_max = 1000, range_nbins = 100,
  xlabel = "lip",
)

SiPixelPhase1TrackingParticleConf = cms.VPSet(
    SiPixelPhase1TrackingParticleMass,
    SiPixelPhase1TrackingParticleCharge,
    SiPixelPhase1TrackingParticleId,
    SiPixelPhase1TrackingParticleNhits,
    SiPixelPhase1TrackingParticleMatched,
    SiPixelPhase1TrackingParticlePt,
    SiPixelPhase1TrackingParticlePhi,
    SiPixelPhase1TrackingParticleEta,
    SiPixelPhase1TrackingParticleVtx,
    SiPixelPhase1TrackingParticleVty,
    SiPixelPhase1TrackingParticleVtz,
    SiPixelPhase1TrackingParticleTip,
    SiPixelPhase1TrackingParticleLip,
)

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
SiPixelPhase1TrackingParticleAnalyzerV = DQMEDAnalyzer('SiPixelPhase1TrackingParticleV',
    src = cms.InputTag("mix","MergedTrackTruth"),
    simHitToken = cms.VInputTag(
                            cms.InputTag( 'g4SimHits','TrackerHitsPixelBarrelLowTof'),
                            cms.InputTag('g4SimHits','TrackerHitsPixelBarrelHighTof'),
                            cms.InputTag('g4SimHits','TrackerHitsPixelEndcapLowTof'),
                            cms.InputTag('g4SimHits','TrackerHitsPixelEndcapHighTof') ),
    histograms = SiPixelPhase1TrackingParticleConf,
    geometry = SiPixelPhase1Geometry
)
from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
premix_stage2.toModify(SiPixelPhase1TrackingParticleAnalyzerV, src = "mixData:MergedTrackTruth")

SiPixelPhase1TrackingParticleHarvesterV = DQMEDHarvester("SiPixelPhase1Harvester",
        histograms = SiPixelPhase1TrackingParticleConf,
        geometry = SiPixelPhase1Geometry
)

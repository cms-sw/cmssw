import FWCore.ParameterSet.Config as cms

generator = cms.EDProducer("RandomXiThetaGunProducer",
  particleId = cms.uint32(2212),

  energy = cms.double(6500),  # nominal beam energy, GeV

  xi_min = cms.double(0.),
  xi_max = cms.double(0.20),
  theta_x_mean = cms.double(0),
  theta_x_sigma = cms.double(50E-6), # in rad
  theta_y_mean = cms.double(0),
  theta_y_sigma = cms.double(50E-6),

  nParticlesSector45 = cms.uint32(1),
  nParticlesSector56 = cms.uint32(1),
)

import FWCore.ParameterSet.Config as cms

ctppsOpticsParameterisation = cms.EDProducer('CTPPSOpticsParameterisation',
    beam1ParticlesTag = cms.InputTag('lhcBeamProducer', 'sector56'),
    beam2ParticlesTag = cms.InputTag('lhcBeamProducer', 'sector45'),

    opticsFileBeam1 = cms.FileInPath('parametrisations/version4-vale1/beam1/parametrization_6500GeV_0p4_185_reco.root'),
    opticsFileBeam2 = cms.FileInPath('parametrisations/version4-vale1/beam2/parametrization_6500GeV_0p4_185_reco.root'),

    # list of detector packages to simulate
    detectorsList = cms.VPSet(
        cms.PSet(
            name = cms.string("RP1"), #FIXME
            scatteringAngle = cms.double(25.e-6), # physics scattering angle, rad
            resolution = cms.double(12.e-6), # RP resolution, m
            minXi = cms.double(0.03),
            maxXi = cms.double(0.17),
        )
    ),
)

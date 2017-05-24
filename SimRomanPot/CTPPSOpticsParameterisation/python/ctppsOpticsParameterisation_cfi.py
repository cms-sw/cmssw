import FWCore.ParameterSet.Config as cms

from SimRomanPot.CTPPSOpticsParameterisation.lhcBeamConditions_cff import lhcBeamConditions_2017PreTS2

ctppsOpticsParameterisation = cms.EDProducer('CTPPSOpticsParameterisation',
    beamConditions = lhcBeamConditions_2017PreTS2,
)

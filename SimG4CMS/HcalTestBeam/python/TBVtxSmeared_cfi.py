import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import *
from SimG4CMS.HcalTestBeam.TBDirectionParameters_cfi import *

VtxSmeared = cms.EDProducer("BeamProfileVtxGenerator",
                            common_beam_direction_parameters,
                            VtxSmearedCommon,
                            BeamMeanX       = cms.double(0.0),
                            BeamMeanY       = cms.double(0.0),
                            BeamSigmaX      = cms.double(0.0001),
                            BeamSigmaY      = cms.double(0.0001),
                            Psi             = cms.double(999.9),
                            GaussianProfile = cms.bool(False),
                            BinX            = cms.int32(50),
                            BinY            = cms.int32(50),
                            File            = cms.string('beam.profile'),
                            UseFile         = cms.bool(False),
                            TimeOffset      = cms.double(0.)
                            )

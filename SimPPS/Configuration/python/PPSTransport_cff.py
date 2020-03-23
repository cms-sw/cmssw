import FWCore.ParameterSet.Config as cms
from SimG4Core.Application.g4SimHits_cfi import *
g4SimHits.Generator.MinEtaCut = cms.double(-13.0)
g4SimHits.Generator.MaxEtaCut = cms.double( 13.0)
g4SimHits.Generator.HepMCProductLabel   = 'LHCTransport'
g4SimHits.SteppingAction.MaxTrackTime = cms.double(2000.0)
g4SimHits.StackingAction.MaxTrackTime = cms.double(2000.0)

from IOMC.RandomEngine.IOMC_cff import *
RandomNumberGeneratorService.LHCTransport.engineName   = cms.untracked.string('TRandom3')

from SimTransport.PPSProtonTransport.TotemTransport_cfi import *

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import Realistic25ns13TeV2016CollisionVtxSmearingParameters 
from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import Realistic25ns13TeVEarly2017CollisionVtxSmearingParameters
from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import Realistic25ns13TeVEarly2018CollisionVtxSmearingParameters

LHCTransport.VtxMeanX  = Realistic25ns13TeV2016CollisionVtxSmearingParameters.X0
LHCTransport.VtxMeanY  = Realistic25ns13TeV2016CollisionVtxSmearingParameters.Y0
LHCTransport.VtxMeanZ  = Realistic25ns13TeV2016CollisionVtxSmearingParameters.Z0

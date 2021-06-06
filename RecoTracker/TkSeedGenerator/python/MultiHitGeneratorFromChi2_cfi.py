import FWCore.ParameterSet.Config as cms

MultiHitGeneratorFromChi2 = cms.PSet(
    ComponentName = cms.string('MultiHitGeneratorFromChi2'),
    maxElement = cms.uint32(100000),
    #fixed phi filtering
    useFixedPreFiltering = cms.bool(False),
    phiPreFiltering = cms.double(0.3),
    #box properties
    extraHitRPhitolerance = cms.double(0.),
    extraHitRZtolerance = cms.double(0.),
    extraZKDBox = cms.double(0.2),
    extraRKDBox = cms.double(0.2),
    extraPhiKDBox = cms.double(0.005),
    fnSigmaRZ = cms.double(2.0),
    #refit&filter hits
    refitHits = cms.bool(True),
    ClusterShapeHitFilterName = cms.string('ClusterShapeHitFilter'),
    TTRHBuilder = cms.string('WithTrackAngle'),
    #chi2 cuts
    maxChi2 = cms.double(5.0),
    chi2VsPtCut = cms.bool(True),
    pt_interv = cms.vdouble(0.4,0.7,1.0,2.0),
    chi2_cuts = cms.vdouble(3.0,4.0,5.0,5.0),
    #debugging
    detIdsToDebug = cms.vint32(0,0,0)
)

from Configuration.Eras.Modifier_peripheralPbPb_cff import peripheralPbPb
from Configuration.Eras.Modifier_pp_on_XeXe_2017_cff import pp_on_XeXe_2017
from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
(peripheralPbPb | pp_on_XeXe_2017 | pp_on_AA).toModify(MultiHitGeneratorFromChi2, maxElement = 1000000)


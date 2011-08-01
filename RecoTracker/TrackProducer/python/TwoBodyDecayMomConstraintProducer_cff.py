import FWCore.ParameterSet.Config as cms

TwoBodyDecayMomConstraint = cms.EDProducer(
    "TwoBodyDecayMomConstraintProducer",
    src = cms.InputTag("AlignmentTrackSelector"),
    beamSpot = cms.InputTag("offlineBeamSpot"),

    ## Define the TBD
    primaryMass = cms.double(91.1876),
    primaryWidth = cms.double(2.4952),
    secondaryMass = cms.double(0.105658),

    ## Configure the TBD estimator
    EstimatorParameters = cms.PSet(
        MaxIterationDifference = cms.untracked.double(0.01),
        RobustificationConstant = cms.untracked.double(1.0),
        MaxIterations = cms.untracked.int32(100),
        UseInvariantMass = cms.untracked.bool(True)
    ),

    ## Cut on chi2 of kinematic fit
    chi2Cut = cms.double( 10. ),

    ## Use momentum at vertex or at innermost surface for constraint? Should have the same
    ## absolute value, but propagation to the innermost surface allows for a position cut.
    momentumForRefitting = cms.string( "atVertex" ),

    ## Fixed error on momentum constraint (to enforce impact on track re-fit)
    fixedMomentumError = cms.double( 1e-4 ),

    ## Configure matching of measured and estimated tsos (only done if 'momentumForRefitting' == 'atInnermostSurface').
    sigmaPositionCut = cms.double( 5e-1 )
)

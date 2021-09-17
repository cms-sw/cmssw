# The following comments couldn't be translated into the new config version:

#also some pairs in the barrel, in case the triplet fail. If the triplet secceds, the pairs should be skipped by the trajectory builder 
#for the TEC we use pairs (Fasthelix not working properly with ss TEC hits)
# genereator for TEC+ outsideIn

#generator for TEC+ insideOut
#generator for TEC- outsideIn
#generator for TEC- insideOut 
import FWCore.ParameterSet.Config as cms

# magnetic field

# cms geometry
# tracker geometry
# tracker numbering
#stripCPE
from RecoLocalTracker.SiStripRecHitConverter.StripCPEfromTrackAngle_cfi import *
#TransientTrackingBuilder
from RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilder_cfi import *
from RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilderWithoutRefit_cfi import *
from RecoTracker.SpecialSeedGenerators.CombinatorialSeedGeneratorForCosmics_cfi import *
import RecoTracker.TkSeedingLayers.seedingLayersEDProducer_cfi as _mod

# seeding layers
combinatorialcosmicseedingtripletsP5 = _mod.seedingLayersEDProducer.clone(
    layerInfo,
    layerList = ['MTOB4+MTOB5+MTOB6', 
                 'MTOB3+MTOB5+MTOB6', 
                 'MTOB3+MTOB4+MTOB5', 
                 'TOB2+MTOB4+MTOB5', 
                 'MTOB3+MTOB4+MTOB6', 
                 'TOB2+MTOB4+MTOB6'],
)
combinatorialcosmicseedingpairsTOBP5 = _mod.seedingLayersEDProducer.clone( 
    layerInfo,
    layerList = ['MTOB5+MTOB6', 
                 'MTOB4+MTOB5'],
)
combinatorialcosmicseedingpairsTECposP5 = _mod.seedingLayersEDProducer.clone(
    layerList = ['TEC1_pos+TEC2_pos', 
                 'TEC2_pos+TEC3_pos', 
                 'TEC3_pos+TEC4_pos', 
                 'TEC4_pos+TEC5_pos', 
                 'TEC5_pos+TEC6_pos', 
                 'TEC6_pos+TEC7_pos', 
                 'TEC7_pos+TEC8_pos', 
                 'TEC8_pos+TEC9_pos'],
    TEC = cms.PSet(
        minRing = cms.int32(5),
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        useRingSlector = cms.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutNone')),
        rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
        maxRing = cms.int32(7)
    )
)
combinatorialcosmicseedingpairsTECnegP5 = _mod.seedingLayersEDProducer.clone(
    layerList = ['TEC1_neg+TEC2_neg', 
                 'TEC2_neg+TEC3_neg', 
                 'TEC3_neg+TEC4_neg', 
                 'TEC4_neg+TEC5_neg', 
                 'TEC5_neg+TEC6_neg', 
                 'TEC6_neg+TEC7_neg', 
                 'TEC7_neg+TEC8_neg', 
                 'TEC8_neg+TEC9_neg'],
    TEC = cms.PSet(
        minRing = cms.int32(5),
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        useRingSlector = cms.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutNone')),
        rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
        maxRing = cms.int32(7)
    )
)
combinatorialcosmicseedinglayersP5Task = cms.Task(combinatorialcosmicseedingtripletsP5, 
                                                  combinatorialcosmicseedingpairsTOBP5,
                                                  combinatorialcosmicseedingpairsTECposP5,
                                                  combinatorialcosmicseedingpairsTECnegP5)
combinatorialcosmicseedinglayersP5 = cms.Sequence(combinatorialcosmicseedinglayersP5Task)
#seeding module
combinatorialcosmicseedfinderP5 = combinatorialcosmicseedfinder.clone(
#replace combinatorialcosmicseedfinderP5.SetMomentum = false
    requireBOFF                = True,
    UseScintillatorsConstraint = False,
    OrderedHitsFactoryPSets    = cms.VPSet(
    cms.PSet(
        ComponentName = cms.string('GenericTripletGenerator'),
        LayerSrc = cms.InputTag("combinatorialcosmicseedingtripletsP5"),
        PropagationDirection = cms.string('alongMomentum'),
        NavigationDirection = cms.string('outsideIn')
    ), 
    cms.PSet(
        ComponentName = cms.string('GenericPairGenerator'),
        LayerSrc = cms.InputTag("combinatorialcosmicseedingpairsTOBP5"),
        PropagationDirection = cms.string('alongMomentum'),
        NavigationDirection = cms.string('outsideIn')
    ), 
    cms.PSet(
        ComponentName = cms.string('GenericPairGenerator'),
        LayerSrc = cms.InputTag("combinatorialcosmicseedingpairsTECposP5"),
        PropagationDirection = cms.string('alongMomentum'),
        NavigationDirection = cms.string('outsideIn')
    ), 
    cms.PSet(
        ComponentName = cms.string('GenericPairGenerator'),
        LayerSrc = cms.InputTag("combinatorialcosmicseedingpairsTECposP5"),
        PropagationDirection = cms.string('alongMomentum'),
        NavigationDirection = cms.string('insideOut')
    ), 
    cms.PSet(
        ComponentName = cms.string('GenericPairGenerator'),
        LayerSrc = cms.InputTag("combinatorialcosmicseedingpairsTECnegP5"),
        PropagationDirection = cms.string('alongMomentum'),
        NavigationDirection = cms.string('outsideIn')
    ), 
    cms.PSet(
        ComponentName = cms.string('GenericPairGenerator'),
        LayerSrc = cms.InputTag("combinatorialcosmicseedingpairsTECnegP5"),
        PropagationDirection = cms.string('alongMomentum'),
        NavigationDirection = cms.string('insideOut')
    ))
)

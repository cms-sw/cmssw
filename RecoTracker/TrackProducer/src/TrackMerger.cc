#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"

#include "RecoTracker/TrackProducer/interface/TrackMerger.h"

#include <boost/foreach.hpp>
#define foreach BOOST_FOREACH

#define TRACK_SORT 1 // just use all hits from inner track, then append from outer outside it
#define DET_SORT   0 // sort hits using global position of detector 
#define HIT_SORT   0 // sort hits using global position of hit


TrackMerger::TrackMerger(const edm::ParameterSet &iConfig) :
    useInnermostState_(iConfig.getParameter<bool>("useInnermostState")),
    debug_(iConfig.getUntrackedParameter<bool>("debug",false)),
    theBuilderName(iConfig.getParameter<std::string>("ttrhBuilderName"))
{
}

TrackMerger::~TrackMerger()
{
}

void TrackMerger::init(const edm::EventSetup &iSetup) 
{
    iSetup.get<TrackerDigiGeometryRecord>().get(theGeometry);
    iSetup.get<IdealMagneticFieldRecord>().get(theMagField);
    iSetup.get<TransientRecHitRecord>().get(theBuilderName,theBuilder);
    iSetup.get<IdealGeometryRecord>().get(theTrkTopo);
}

TrackCandidate TrackMerger::merge(const reco::Track &inner, const reco::Track &outer) const 
{
    std::vector<const TrackingRecHit *> hits;
    hits.reserve(inner.recHitsSize() + outer.recHitsSize());
    if (debug_) std::cout << "Inner track hits: " << std::endl;
    for (trackingRecHit_iterator it = inner.recHitsBegin(), ed = inner.recHitsEnd(); it != ed; ++it) {
        hits.push_back(&**it);
        if (debug_) {
            DetId id(hits.back()->geographicalId());
            std::cout << "   subdet " << id.subdetId() << "  layer " << theTrkTopo->layer(id) << " valid " << hits.back()->isValid() << "   detid: " << id() << std::endl;
        }
    }
    if (debug_) std::cout << "Outer track hits: " << std::endl;

#if TRACK_SORT
    DetId lastId(hits.back()->geographicalId());
    int lastSubdet = lastId.subdetId(); 
    unsigned int lastLayer = theTrkTopo->layer(lastId);
    for (trackingRecHit_iterator it = outer.recHitsBegin(), ed = outer.recHitsEnd(); it != ed; ++it) {
        const TrackingRecHit *hit = &**it;
        DetId id(hit->geographicalId());
        int thisSubdet = id.subdetId();
        if (thisSubdet > lastSubdet || (thisSubdet == lastSubdet && theTrkTopo->layer(id) > lastLayer)) {
            hits.push_back(hit);
            if (debug_) std::cout << "  adding   subdet " << id.subdetId() << "  layer " << theTrkTopo->layer(id) << " valid " << hit->isValid() << "   detid: " << id() << std::endl;
        } else {
            if (debug_) std::cout << "  skipping subdet " << thisSubdet << "  layer " << theTrkTopo->layer(id) << " valid " << hit->isValid() << "   detid: " << id() << std::endl;
        }
    }
#else
    size_t nHitsFirstTrack = hits.size();
    for (trackingRecHit_iterator it = outer.recHitsBegin(), ed = outer.recHitsEnd(); it != ed; ++it) {
        const TrackingRecHit *hit = &**it;
        DetId id(hit->geographicalId());
        int  lay = theTrkTopo->layer(id);
        bool shared = false;
        bool valid  = hit->isValid();
        if (debug_) std::cout << "   subdet " << id.subdetId() << "  layer " << theTrkTopo->layer(id) << " valid " << valid << "   detid: " << id() << std::endl;
        size_t iHit = 0;
        foreach(const TrackingRecHit *& hit2, hits) {
            ++iHit; if (iHit >  nHitsFirstTrack) break;
            DetId id2 = hit2->geographicalId();
            if (id.subdetId() != id2.subdetId()) continue;
            if (theTrkTopo->layer(id2) != lay) continue;
            if (hit->sharesInput(hit2, TrackingRecHit::all)) { 
                if (debug_) std::cout << "        discared as duplicate of other hit" << id() << std::endl;
                shared = true; break; 
            }
            if (hit2->isValid() && !valid) { 
                if (debug_) std::cout << "        replacing old invalid hit on detid " << id2() << std::endl;
                hit2 = hit; shared = true; break; 
            }
            if (debug_) std::cout << "        discared as additional hit on layer that already contains hit with detid " << id() << std::endl;
            shared = true; break;
        }
        if (shared) continue;
        hits.push_back(hit);
    }
#endif
    
    math::XYZVector p = (inner.innerMomentum() + outer.outerMomentum());
    GlobalVector v(p.x(), p.y(), p.z());
    if (!useInnermostState_) v *= -1;

    TrackCandidate::RecHitContainer ownHits;
    unsigned int nhits = hits.size();
    ownHits.reserve(nhits);

#if TRACK_SORT
    if (!useInnermostState_) std::reverse(hits.begin(), hits.end());
    foreach(const TrackingRecHit * hit, hits) {
        ownHits.push_back(*hit);
    }
#elif DET_SORT
    // OLD METHOD, sometimes fails
    std::sort(hits.begin(), hits.end(), MomentumSort(v, &*theGeometry));

    foreach(const TrackingRecHit * hit, hits) {
        ownHits.push_back(*hit);
    }
#else
    // NEW sort, more accurate
    std::vector<TransientTrackingRecHit::RecHitPointer> ttrh(nhits);
    for (unsigned int i = 0; i < nhits; ++i) ttrh[i] = theBuilder->build(hits[i]);
    std::sort(ttrh.begin(), ttrh.end(), GlobalMomentumSort(v));

    foreach(const TransientTrackingRecHit::RecHitPointer & hit, ttrh) {
        ownHits.push_back(*hit->hit());
    }
#endif

    PTrajectoryStateOnDet state;
    PropagationDirection pdir;
    if (useInnermostState_) {
        pdir = alongMomentum;
        if ((inner.outerPosition()-inner.innerPosition()).Dot(inner.momentum()) >= 0) {
            // use inner state
            TrajectoryStateOnSurface originalTsosIn(trajectoryStateTransform::innerStateOnSurface(inner, *theGeometry, &*theMagField));
            state = trajectoryStateTransform::persistentState( originalTsosIn, DetId(inner.innerDetId()) );
        } else { 
            // use outer state
            TrajectoryStateOnSurface originalTsosOut(trajectoryStateTransform::outerStateOnSurface(inner, *theGeometry, &*theMagField));
            state = trajectoryStateTransform::persistentState( originalTsosOut, DetId(inner.outerDetId()) );
        }
    } else {
        pdir = oppositeToMomentum;
        if ((outer.outerPosition()-inner.innerPosition()).Dot(inner.momentum()) >= 0) {
            // use outer state
            TrajectoryStateOnSurface originalTsosOut(trajectoryStateTransform::outerStateOnSurface(outer, *theGeometry, &*theMagField));
            state = trajectoryStateTransform::persistentState( originalTsosOut, DetId(outer.outerDetId()) );
        } else { 
            // use inner state
            TrajectoryStateOnSurface originalTsosIn(trajectoryStateTransform::innerStateOnSurface(outer, *theGeometry, &*theMagField));
            state = trajectoryStateTransform::persistentState( originalTsosIn, DetId(outer.innerDetId()) );
        }

    }
     TrajectorySeed seed(state, TrackCandidate::RecHitContainer(), pdir);
    return TrackCandidate(ownHits, seed, state, (useInnermostState_ ? inner : outer).seedRef());
}


bool TrackMerger::MomentumSort::operator()(const TrackingRecHit *hit1, const TrackingRecHit *hit2) const 
{
    const GeomDet *det1 = geom_->idToDet(hit1->geographicalId());
    const GeomDet *det2 = geom_->idToDet(hit2->geographicalId());
    GlobalPoint p1 = det1->toGlobal(LocalPoint(0,0,0));
    GlobalPoint p2 = det2->toGlobal(LocalPoint(0,0,0));
    return (p2 - p1).dot(dir_) > 0;
}

bool TrackMerger::GlobalMomentumSort::operator()(const TransientTrackingRecHit::RecHitPointer &hit1, const TransientTrackingRecHit::RecHitPointer &hit2) const 
{
    GlobalPoint p1 = hit1->isValid() ? hit1->globalPosition() : hit1->det()->position();
    GlobalPoint p2 = hit2->isValid() ? hit2->globalPosition() : hit2->det()->position();
    return (p2 - p1).dot(dir_) > 0;
}

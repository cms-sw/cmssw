#include "RecoTracker/TrackProducer/interface/TrackProducerAlgorithm.h"

#include "DataFormats/Common/interface/OrphanHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidate.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"

#include "TrackingTools/TrackFitters/interface/TrajectoryFitter.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"
#include "TrackingTools/PatternTools/interface/TransverseImpactPointExtrapolator.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "RecoTracker/TrackProducer/interface/TrackingRecHitLessFromGlobalPosition.h"

#include "TrackingTools/PatternTools/interface/TSCBLBuilderNoMaterial.h"
#include "Utilities/General/interface/CMSexception.h"

#include "RecoTracker/TransientTrackingRecHit/interface/TRecHit2DPosConstraint.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TRecHit1DMomConstraint.h"
// #include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/ErrorFrameTransformer.h"
#include "TrackingTools/TrackFitters/interface/RecHitSorter.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"

namespace {
#ifdef STAT_TSB
  struct StatCount {
    long long totTrack;
    long long totLoop;
    long long totGsfTrack;
    void zero() {
      totTrack=totLoop=totGsfTrack=0;
    }
    void track(int l) {
      if (l>0) ++totLoop; else ++totTrack;
    }
    void gsf() {++totGsfTrack;}


    void print() const {
      std::cout << "TrackProducer stat\nTrack/Loop/Gsf "
    		<<  totTrack <<'/'<< totLoop <<'/'<< totGsfTrack
		<< std::endl;
    }
    StatCount() { zero();}
    ~StatCount() { print();}
  };

#else
  struct StatCount {
    void track(int){}
    void gsf(){}
  };
#endif

  StatCount statCount;

}




template <> bool
TrackProducerAlgorithm<reco::Track>::buildTrack (const TrajectoryFitter * theFitter,
						 const Propagator * thePropagator,
						 AlgoProductCollection& algoResults,
						 TransientTrackingRecHit::RecHitContainer& hits,
						 TrajectoryStateOnSurface& theTSOS,
						 const TrajectorySeed& seed,
						 float ndof,
						 const reco::BeamSpot& bs,
						 SeedRef seedRef,
						 int qualityMask,signed char nLoops)						 
{
  //variable declarations
  reco::Track * theTrack;
  Trajectory * theTraj; 
  PropagationDirection seedDir = seed.direction();
      
  //perform the fit: the result's size is 1 if it succeded, 0 if fails
  Trajectory && trajTmp = theFitter->fitOne(seed, hits, theTSOS,(nLoops>0) ? TrajectoryFitter::looper : TrajectoryFitter::standard);
  if unlikely(!trajTmp.isValid()) return false;
  
  
  
  theTraj = new Trajectory(std::move(trajTmp));
  theTraj->setSeedRef(seedRef);
  
  // TrajectoryStateOnSurface innertsos;
  // if (theTraj->direction() == alongMomentum) {
  //  innertsos = theTraj->firstMeasurement().updatedState();
  // } else { 
  //  innertsos = theTraj->lastMeasurement().updatedState();
  // }
  
  ndof = 0;
  for (auto const & tm : theTraj->measurements()) {
    auto const & h = tm.recHitR();
    if (h.isValid()) ndof = ndof + float(h.dimension())*h.weight();  // two virtual calls!
  }
  
  ndof -= 5.f;
  if unlikely(std::abs(theTSOS.magneticField()->nominalValue())<DBL_MIN) ++ndof;  // same as -4
 
 
  //    const FreeTrajectoryState & stateForProjectionToBeamLine=*innertsos.freeState();
  const TrajectoryStateOnSurface & stateForProjectionToBeamLineOnSurface = theTraj->closestMeasurement(GlobalPoint(bs.x0(),bs.y0(),bs.z0())).updatedState();

  if unlikely(!stateForProjectionToBeamLineOnSurface.isValid()){
    edm::LogError("CannotPropagateToBeamLine")<<"the state on the closest measurement isnot valid. skipping track.";
    delete theTraj;
    return false;
  }
  const FreeTrajectoryState & stateForProjectionToBeamLine=*stateForProjectionToBeamLineOnSurface.freeState();
  
  LogDebug("TrackProducer") << "stateForProjectionToBeamLine=" << stateForProjectionToBeamLine;
  
  TSCBLBuilderNoMaterial tscblBuilder;
  TrajectoryStateClosestToBeamLine tscbl = tscblBuilder(stateForProjectionToBeamLine,bs);
  
  if unlikely(!tscbl.isValid()) {
    delete theTraj;
    return false;
  }
  
  GlobalPoint v = tscbl.trackStateAtPCA().position();
  math::XYZPoint  pos( v.x(), v.y(), v.z() );
  GlobalVector p = tscbl.trackStateAtPCA().momentum();
  math::XYZVector mom( p.x(), p.y(), p.z() );
  
  LogDebug("TrackProducer") << "pos=" << v << " mom=" << p << " pt=" << p.perp() << " mag=" << p.mag();
  
  theTrack = new reco::Track(theTraj->chiSquared(),
			     int(ndof),//FIXME fix weight() in TrackingRecHit
			     pos, mom, tscbl.trackStateAtPCA().charge(), 
			     tscbl.trackStateAtPCA().curvilinearError(),
			     algo_);
  
  theTrack->setQualityMask(qualityMask);
  theTrack->setNLoops(nLoops);
  
  LogDebug("TrackProducer") << "theTrack->pt()=" << theTrack->pt();
  
  LogDebug("TrackProducer") <<"track done\n";
  
  AlgoProduct aProduct(theTraj,std::make_pair(theTrack,seedDir));
  algoResults.push_back(aProduct);
  
  statCount.track(nLoops);

  return true;
} 

template <> bool
TrackProducerAlgorithm<reco::GsfTrack>::buildTrack (const TrajectoryFitter * theFitter,
						    const Propagator * thePropagator,
						    AlgoProductCollection& algoResults,
						    TransientTrackingRecHit::RecHitContainer& hits,
						    TrajectoryStateOnSurface& theTSOS,
						    const TrajectorySeed& seed,
						    float ndof,
						    const reco::BeamSpot& bs,
						    SeedRef seedRef,
						    int qualityMask,signed char nLoops)
{
  //variable declarations
  reco::GsfTrack * theTrack;
  Trajectory * theTraj; 
  PropagationDirection seedDir = seed.direction();
  
  Trajectory && trajTmp = theFitter->fitOne(seed, hits, theTSOS,(nLoops>0) ? TrajectoryFitter::looper: TrajectoryFitter::standard);
  if unlikely(!trajTmp.isValid()) return false;
  
  
  theTraj = new Trajectory( std::move(trajTmp) );
  theTraj->setSeedRef(seedRef);
  
  //  TrajectoryStateOnSurface innertsos;
  // TrajectoryStateOnSurface outertsos;

  // if (theTraj->direction() == alongMomentum) {
  //  innertsos = theTraj->firstMeasurement().updatedState();
  //  outertsos = theTraj->lastMeasurement().updatedState();
  // } else { 
  //  innertsos = theTraj->lastMeasurement().updatedState();
  //  outertsos = theTraj->firstMeasurement().updatedState();
  // }
  //     std::cout
  //       << "Nr. of first / last states = "
  //       << innertsos.components().size() << " "
  //       << outertsos.components().size() << std::endl;
  //     std::vector<TrajectoryStateOnSurface> components = 
  //       innertsos.components();
  //     double sinTheta = 
  //       sin(innertsos.globalMomentum().theta());
  //     for ( std::vector<TrajectoryStateOnSurface>::const_iterator ic=components.begin();
  // 	  ic!=components.end(); ic++ ) {
  //       std::cout << " comp " << ic-components.begin() << " "
  // 		<< (*ic).weight() << " "
  // 		<< (*ic).localParameters().vector()[0]/sinTheta << " "
  // 		<< sqrt((*ic).localError().matrix()[0][0])/sinTheta << std::endl;
  //     }
  
  ndof = 0;
  for (auto const & tm : theTraj->measurements()) {
    auto const & h = tm.recHitR();
    if (h.isValid()) ndof = ndof + h.dimension()*h.weight();
  }
  
  ndof = ndof - 5;
  if unlikely(std::abs(theTSOS.magneticField()->nominalValue())<DBL_MIN) ++ndof;  // same as -4
  
  
  //    const FreeTrajectoryState & stateForProjectionToBeamLine=*innertsos.freeState();
  const TrajectoryStateOnSurface & stateForProjectionToBeamLineOnSurface = theTraj->closestMeasurement(GlobalPoint(bs.x0(),bs.y0(),bs.z0())).updatedState();
  if unlikely(!stateForProjectionToBeamLineOnSurface.isValid()){
      edm::LogError("CannotPropagateToBeamLine")<<"the state on the closest measurement isnot valid. skipping track.";
      delete theTraj;
      return false;
    }    
  
  const FreeTrajectoryState & stateForProjectionToBeamLine=*theTraj->closestMeasurement(GlobalPoint(bs.x0(),bs.y0(),bs.z0())).updatedState().freeState();
  
  LogDebug("GsfTrackProducer") << "stateForProjectionToBeamLine=" << stateForProjectionToBeamLine;
  
  TSCBLBuilderNoMaterial tscblBuilder;
  TrajectoryStateClosestToBeamLine tscbl = tscblBuilder(stateForProjectionToBeamLine,bs);
  
  if unlikely(tscbl.isValid()==false) {
      delete theTraj;
      return false;
    }
  
  GlobalPoint v = tscbl.trackStateAtPCA().position();
  math::XYZPoint  pos( v.x(), v.y(), v.z() );
  GlobalVector p = tscbl.trackStateAtPCA().momentum();
  math::XYZVector mom( p.x(), p.y(), p.z() );
  
  LogDebug("GsfTrackProducer") << "pos=" << v << " mom=" << p << " pt=" << p.perp() << " mag=" << p.mag();
  
  theTrack = new reco::GsfTrack(theTraj->chiSquared(),
				int(ndof),//FIXME fix weight() in TrackingRecHit
				//			       theTraj->foundHits(),//FIXME to be fixed in Trajectory.h
				//			       0, //FIXME no corresponding method in trajectory.h
				//			       theTraj->lostHits(),//FIXME to be fixed in Trajectory.h
				pos, mom, tscbl.trackStateAtPCA().charge(), tscbl.trackStateAtPCA().curvilinearError());    
  theTrack->setAlgorithm(algo_);
  
  LogDebug("GsfTrackProducer") <<"track done\n";
  
  AlgoProduct aProduct(theTraj,std::make_pair(theTrack,seedDir));
  LogDebug("GsfTrackProducer") <<"track done1\n";
  algoResults.push_back(aProduct);
  LogDebug("GsfTrackProducer") <<"track done2\n";
  
  statCount.gsf();
  return true;
} 

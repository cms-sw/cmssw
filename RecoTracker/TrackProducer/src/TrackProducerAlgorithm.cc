#include <sstream>

#include "DataFormats/Common/interface/OrphanHandle.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/ErrorFrameTransformer.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidate.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"
#include "DataFormats/TrackerRecHit2D/interface/TrackingRecHitLessFromGlobalPosition.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "RecoTracker/TrackProducer/interface/TrackProducerAlgorithm.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TRecHit1DMomConstraint.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TRecHit2DPosConstraint.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"
#include "TrackingTools/PatternTools/interface/TSCBLBuilderNoMaterial.h"
#include "TrackingTools/PatternTools/interface/TSCBLBuilderWithPropagator.h"
#include "TrackingTools/PatternTools/interface/TransverseImpactPointExtrapolator.h"
#include "TrackingTools/TrackFitters/interface/RecHitSorter.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryFitter.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"

// #define VI_DEBUG
// #define STAT_TSB

#ifdef VI_DEBUG
#define DPRINT(x) std::cout << x << ": "
#else
#define DPRINT(x) LogTrace(x)
#endif
  

namespace {
#ifdef STAT_TSB
  struct StatCount {
    long long totTrack=0;
    long long totLoop=0;
    long long totGsfTrack=0;
    long long totFound=0;
    long long totLost=0;
    long long totAlgo[15];
    void track(int l) {
      if (l>0) ++totLoop; else ++totTrack;
    }
    void hits(int f, int l) { totFound+=f; totLost+=l;} 
    void gsf() {++totGsfTrack;}
    void algo(int a) { if (a>=0 && a<15) ++totAlgo[a];}


    void print() const {
      std::cout << "TrackProducer stat\nTrack/Loop/Gsf/FoundHits/LostHits//algos "
    		<<  totTrack <<'/'<< totLoop <<'/'<< totGsfTrack  <<'/'<< totFound  <<'/'<< totLost<<'/';
      for (auto a : totAlgo) std::cout << '/'<< a;
	std::cout  << std::endl;
    }
    StatCount() {}
    ~StatCount() { print();}
  };
  StatCount statCount;

#else
  struct StatCount {
    void track(int){}
    void hits(int, int){}
    void gsf(){}
    void algo(int){}
  };
  CMS_THREAD_SAFE StatCount statCount;
#endif


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

  PropagationDirection seedDir = seed.direction();
      
  //perform the fit: the result's size is 1 if it succeded, 0 if fails
  Trajectory && trajTmp = theFitter->fitOne(seed, hits, theTSOS,(nLoops>0) ? TrajectoryFitter::looper : TrajectoryFitter::standard);
  if UNLIKELY(!trajTmp.isValid()) {
     DPRINT("TrackFitters") << "fit failed " << algo_ << ": " <<  hits.size() <<'|' << int(nLoops) << ' ' << std::endl; 
     return false;
  }
  
  
  auto theTraj = new Trajectory(std::move(trajTmp));
  theTraj->setSeedRef(seedRef);
  
  statCount.hits(theTraj->foundHits(),theTraj->lostHits());
  statCount.algo(int(algo_));

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
  if UNLIKELY(std::abs(theTSOS.magneticField()->nominalValue())<DBL_MIN) ++ndof;  // same as -4
 

#if defined(VI_DEBUG) || defined(EDM_ML_DEBUG)
int chit[7]={};
int kk=0;
for (auto const & tm : theTraj->measurements()) {
  ++kk;
  auto const & hit = tm.recHitR();
  if (!hit.isValid()) ++chit[0];
  if (hit.det()==nullptr) ++chit[1];
  if ( trackerHitRTTI::isUndef(hit) ) continue;
  if(0) std::cout << "h " << kk << ": "<< hit.localPosition() << ' ' << hit.localPositionError() << ' ' << tm.estimate() << std::endl;
  if ( hit.dimension()!=2 ) {
    ++chit[2];
  } else {
    auto const & thit = static_cast<BaseTrackerRecHit const&>(hit);
    auto const & clus = thit.firstClusterRef();
    if (clus.isPixel()) ++chit[3];
    else if (thit.isMatched()) {
      ++chit[4];
    } else  if (thit.isProjected()) {
      ++chit[5];
    } else {
      ++chit[6];
        }
  }
 }

   std::ostringstream ss;
   ss << algo_ << ": " <<  hits.size() <<'|' <<theTraj->measurements().size()<<'|' << int(nLoops) << ' ';   for (auto c:chit) ss << c <<'/'; ss << std::endl;
   DPRINT("TrackProducer") << ss.str();

#endif
 
  //if geometricInnerState_ is false the state for projection to beam line is the state attached to the first hit: to be used for loopers
  //if geometricInnerState_ is true the state for projection to beam line is the one from the (geometrically) closest measurement to the beam line: to be sued for non-collision tracks
  //the two shouuld give the same result for collision tracks that are NOT loopers
  TrajectoryStateOnSurface stateForProjectionToBeamLineOnSurface;
  if (geometricInnerState_) {
    stateForProjectionToBeamLineOnSurface = theTraj->closestMeasurement(GlobalPoint(bs.x0(),bs.y0(),bs.z0())).updatedState();
  } else {
    if (theTraj->direction() == alongMomentum) {
      stateForProjectionToBeamLineOnSurface = theTraj->firstMeasurement().updatedState();
    } else { 
      stateForProjectionToBeamLineOnSurface = theTraj->lastMeasurement().updatedState();
    }
  }

  if UNLIKELY(!stateForProjectionToBeamLineOnSurface.isValid()){
    edm::LogError("CannotPropagateToBeamLine")<<"the state on the closest measurement isnot valid. skipping track.";
    delete theTraj;
    return false;
  }
  const FreeTrajectoryState & stateForProjectionToBeamLine=*stateForProjectionToBeamLineOnSurface.freeState();
  
  LogDebug("TrackProducer") << "stateForProjectionToBeamLine=" << stateForProjectionToBeamLine;
  
//  TSCBLBuilderNoMaterial tscblBuilder;
//  TrajectoryStateClosestToBeamLine tscbl = tscblBuilder(stateForProjectionToBeamLine,bs);

  TrajectoryStateClosestToBeamLine tscbl;
  if (usePropagatorForPCA_){
    //std::cout << "PROPAGATOR FOR PCA" << std::endl;
    TSCBLBuilderWithPropagator tscblBuilder(*thePropagator);
    tscbl = tscblBuilder(stateForProjectionToBeamLine,bs);
  } else {
    TSCBLBuilderNoMaterial tscblBuilder;
    tscbl = tscblBuilder(stateForProjectionToBeamLine,bs);
  }

  
  if UNLIKELY(!tscbl.isValid()) {
    delete theTraj;
    return false;
  }
  
  GlobalPoint v = tscbl.trackStateAtPCA().position();
  math::XYZPoint  pos( v.x(), v.y(), v.z() );
  GlobalVector p = tscbl.trackStateAtPCA().momentum();
  math::XYZVector mom( p.x(), p.y(), p.z() );
  
  LogDebug("TrackProducer") << "pos=" << v << " mom=" << p << " pt=" << p.perp() << " mag=" << p.mag();
  
  auto theTrack = new reco::Track(theTraj->chiSquared(),
			     int(ndof),//FIXME fix weight() in TrackingRecHit
			     pos, mom, tscbl.trackStateAtPCA().charge(), 
			     tscbl.trackStateAtPCA().curvilinearError(),
			     algo_);
  
  if(originalAlgo_ != reco::TrackBase::undefAlgorithm) theTrack->setOriginalAlgorithm(originalAlgo_);
  if(algoMask_.any())                                  theTrack->setAlgoMask(algoMask_);
  theTrack->setQualityMask(qualityMask);
  theTrack->setNLoops(nLoops);
  theTrack->setStopReason(stopReason_);

  LogDebug("TrackProducer") << "theTrack->pt()=" << theTrack->pt();
  
  LogDebug("TrackProducer") <<"track done\n";
  
  AlgoProduct aProduct{theTraj,theTrack,seedDir,0};
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

  PropagationDirection seedDir = seed.direction();
  
  Trajectory && trajTmp = theFitter->fitOne(seed, hits, theTSOS,(nLoops>0) ? TrajectoryFitter::looper: TrajectoryFitter::standard);
  if UNLIKELY(!trajTmp.isValid()) return false;
  
  
  auto theTraj = new Trajectory( std::move(trajTmp) );
  theTraj->setSeedRef(seedRef);

#ifdef EDM_ML_DEBUG  
  TrajectoryStateOnSurface innertsos;
  TrajectoryStateOnSurface outertsos;

  if (theTraj->direction() == alongMomentum) {
    innertsos = theTraj->firstMeasurement().updatedState();
    outertsos = theTraj->lastMeasurement().updatedState();
  } else { 
    innertsos = theTraj->lastMeasurement().updatedState();
     outertsos = theTraj->firstMeasurement().updatedState();
  }
  std::ostringstream ss;
  auto dc = [&](TrajectoryStateOnSurface const & tsos){ 
     std::vector<TrajectoryStateOnSurface> const & components = tsos.components();
     auto sinTheta =  std::sin(tsos.globalMomentum().theta());
     for (auto const & ic : components) ss << ic.weight() << "/"; ss << "\n";
     for (auto const & ic : components) ss << ic.localParameters().vector()[0]/sinTheta << "/"; ss << "\n";
     for (auto const & ic : components) ss << std::sqrt(ic.localError().matrix()(0,0))/sinTheta << "/"; 
  };
  ss  << "\ninner comps\n";
  dc(innertsos);
  ss  << "\nouter comps\n";
  dc(outertsos);
  LogDebug("TrackProducer")
 	   << "Nr. of first / last states = "
  	   << innertsos.components().size() << " "
           << outertsos.components().size() << ss.str();
#endif  

  ndof = 0;
  for (auto const & tm : theTraj->measurements()) {
    auto const & h = tm.recHitR();
    if (h.isValid()) ndof = ndof + h.dimension()*h.weight();
  }
  
  ndof = ndof - 5;
  if UNLIKELY(std::abs(theTSOS.magneticField()->nominalValue())<DBL_MIN) ++ndof;  // same as -4
  
  
  //if geometricInnerState_ is false the state for projection to beam line is the state attached to the first hit: to be used for loopers
  //if geometricInnerState_ is true the state for projection to beam line is the one from the (geometrically) closest measurement to the beam line: to be sued for non-collision tracks
  //the two shouuld give the same result for collision tracks that are NOT loopers
  TrajectoryStateOnSurface stateForProjectionToBeamLineOnSurface;
  if (geometricInnerState_) {
    stateForProjectionToBeamLineOnSurface = theTraj->closestMeasurement(GlobalPoint(bs.x0(),bs.y0(),bs.z0())).updatedState();
  } else {
    if (theTraj->direction() == alongMomentum) {
      stateForProjectionToBeamLineOnSurface = theTraj->firstMeasurement().updatedState();
    } else { 
      stateForProjectionToBeamLineOnSurface = theTraj->lastMeasurement().updatedState();
    }
  }

  if UNLIKELY(!stateForProjectionToBeamLineOnSurface.isValid()){
      edm::LogError("CannotPropagateToBeamLine")<<"the state on the closest measurement isnot valid. skipping track.";
      delete theTraj;
      return false;
    }    
  
  const FreeTrajectoryState & stateForProjectionToBeamLine=*stateForProjectionToBeamLineOnSurface.freeState();
  
  LogDebug("GsfTrackProducer") << "stateForProjectionToBeamLine=" << stateForProjectionToBeamLine;
  
//  TSCBLBuilderNoMaterial tscblBuilder;
//  TrajectoryStateClosestToBeamLine tscbl = tscblBuilder(stateForProjectionToBeamLine,bs);

  TrajectoryStateClosestToBeamLine tscbl;
  if (usePropagatorForPCA_){
    TSCBLBuilderWithPropagator tscblBuilder(*thePropagator);
    tscbl = tscblBuilder(stateForProjectionToBeamLine,bs);    
  } else {
    TSCBLBuilderNoMaterial tscblBuilder;
    tscbl = tscblBuilder(stateForProjectionToBeamLine,bs);
  }  

  
  if UNLIKELY(tscbl.isValid()==false) {
      delete theTraj;
      return false;
    }
  
  GlobalPoint v = tscbl.trackStateAtPCA().position();
  math::XYZPoint  pos( v.x(), v.y(), v.z() );
  GlobalVector p = tscbl.trackStateAtPCA().momentum();
  math::XYZVector mom( p.x(), p.y(), p.z() );
  
  LogDebug("GsfTrackProducer") << "pos=" << v << " mom=" << p << " pt=" << p.perp() << " mag=" << p.mag();
  
  auto theTrack = new reco::GsfTrack(theTraj->chiSquared(),
				int(ndof),//FIXME fix weight() in TrackingRecHit
				//			       theTraj->foundHits(),//FIXME to be fixed in Trajectory.h
				//			       0, //FIXME no corresponding method in trajectory.h
				//			       theTraj->lostHits(),//FIXME to be fixed in Trajectory.h
				pos, mom, tscbl.trackStateAtPCA().charge(), tscbl.trackStateAtPCA().curvilinearError());    
  theTrack->setAlgorithm(algo_);
  if(originalAlgo_ != reco::TrackBase::undefAlgorithm) theTrack->setOriginalAlgorithm(originalAlgo_);
  if(algoMask_.any())                                  theTrack->setAlgoMask(algoMask_);

  theTrack->setStopReason(stopReason_);

  LogDebug("GsfTrackProducer") <<"track done\n";
  
  AlgoProduct aProduct{theTraj,theTrack,seedDir,0};

  LogDebug("GsfTrackProducer") <<"track done1\n";
  algoResults.push_back(aProduct);
  LogDebug("GsfTrackProducer") <<"track done2\n";
  
  statCount.gsf();
  return true;
} 

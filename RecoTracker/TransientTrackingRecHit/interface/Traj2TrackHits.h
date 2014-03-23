#ifndef Traj2TrackHits_H
#define Traj2TrackHits_H

#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"


#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1D.h"

#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/StripClusterParameterEstimator.h"
#include "Geometry/TrackerGeometryBuilder/interface/GluedGeomDet.h"
#include <Geometry/CommonDetUnit/interface/GeomDetType.h>


class Traj2TrackHits {
private:
  const StripClusterParameterEstimator * theCPE;
  bool keepOrder;

public:

  explicit Traj2TrackHits(const TransientTrackingRecHitBuilder* builder,bool ikeepOrder) :
    theCPE(static_cast<TkTransientTrackingRecHitBuilder const *>(builder)->stripClusterParameterEstimator()),
    keepOrder(ikeepOrder){}

  void operator()(Trajectory const & traj, TrackingRecHitCollection & hits, bool splitting) const {
    // ---  NOTA BENE: the convention is to sort hits and measurements "along the momentum".
    bool along = traj.direction() == alongMomentum;
    auto const & meas = traj.measurements();
    hits.reserve(meas.size());
    if(!splitting){
      if (keepOrder | along) copy(meas.begin(),meas.end(),hits);
      else copy(meas.rbegin(),meas.rend(),hits);
      return;
    }
    if (keepOrder | along) split(meas.begin(),meas.end(),hits, along);
    else split(meas.rbegin(),meas.rend(),hits,along);
  }

private:
  template<typename HI>
  static void copy(HI itm, HI e, TrackingRecHitCollection & hits) { 
    for(;itm!=e;++itm) hits.push_back((*itm).recHit()->hit()->clone());
  }

  template<typename HI>
  void split(HI itm, HI e, TrackingRecHitCollection & hits, bool along) const { 
    for(;itm!=e;++itm) {
      auto const & hit = *(*itm).recHit()->hit();
      if(trackerHitRTTI::isUndef(hit) | ( hit.dimension()!=2) ) {
	hits.push_back(hit.clone());
	continue;
      }
      auto const & thit = static_cast<BaseTrackerRecHit const&>(hit);
      auto const & clus = thit.firstClusterRef();
      if (clus.isPixel()) hits.push_back(hit.clone());
      else if (thit.isMatched()) {
	auto zdir = itm->updatedState().localDirection().z();
	if (keepOrder & (!along)) zdir = -zdir;
	split(*itm,static_cast<SiStripMatchedRecHit2D const&>(thit),hits,zdir);
      }else  if (thit.isProjected()) {
	auto detU = static_cast<ProjectedSiStripRecHit2D const&>(thit).originalDet();
	hits.push_back(build(*detU, clus));
      } else hits.push_back(clone(thit));
    }
  }

  TrackingRecHit * clone(BaseTrackerRecHit const & hit2D ) const {
    auto const & detU = *hit2D.detUnit();
    //Use 2D SiStripRecHit in endcap
    bool endcap = detU.type().isEndcap();
    if (endcap) return hit2D.clone();
    return new SiStripRecHit1D(hit2D.localPosition(),
			       LocalError(hit2D.localPositionError().xx(),0.f,std::numeric_limits<float>::max()),
			       *hit2D.det(), hit2D.firstClusterRef());

  }


  BaseTrackerRecHit * build(GeomDetUnit const & idet,
			    OmniClusterRef const &   clus) const {
    //Use 2D SiStripRecHit in endcap
    bool endcap = idet.type().isEndcap();
    auto && lv = theCPE->localParameters(clus.stripCluster(),idet);
    if (endcap) return new SiStripRecHit2D(lv.first,lv.second,idet,clus);
    return new SiStripRecHit1D(lv.first, LocalError(lv.second.xx(),0.f,std::numeric_limits<float>::max()),idet,clus);
  }

  void split(TrajectoryMeasurement const & itm, 
	     SiStripMatchedRecHit2D const& mhit, TrackingRecHitCollection & hits, float zdir) const {
    const GluedGeomDet *gdet = static_cast<const GluedGeomDet *> (mhit.det());
        
    auto hitM = build (*gdet->monoDet(),
		       mhit.monoClusterRef());
    auto hitS = build(*gdet->stereoDet(),
		      mhit.stereoClusterRef());

    // we should find a faster way
    LocalPoint firstLocalPos = 
      itm.updatedState().surface().toLocal(gdet->monoDet()->position());	
    LocalPoint secondLocalPos = 
      itm.updatedState().surface().toLocal(gdet->stereoDet()->position());
    LocalVector Delta = secondLocalPos - firstLocalPos;
    float scalar  = Delta.z() * zdir;
    // hit along the direction
    if(scalar<0) {
      hits.push_back(hitS);
      hits.push_back(hitM);
    } else {
      hits.push_back(hitM);
      hits.push_back(hitS);
    }

  }
    

};


#endif

#include "RecoTracker/TransientTrackingRecHit/interface/TkClonerImpl.h"

#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1D.h"
#include "DataFormats/TrackerRecHit2D/interface/Phase2TrackerRecHit1D.h"

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"


#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/PixelClusterParameterEstimator.h"

#include "Geometry/CommonDetUnit/interface/GluedGeomDet.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/StripClusterParameterEstimator.h"
#include "RecoLocalTracker/SiStripRecHitConverter/interface/SiStripRecHitMatcher.h"

#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"

#include<iostream>
#include <memory>

std::unique_ptr<SiPixelRecHit>  TkClonerImpl::operator()(SiPixelRecHit const & hit, TrajectoryStateOnSurface const& tsos) const {
  const SiPixelCluster& clust = *hit.cluster();  
  auto && params = pixelCPE->getParameters( clust, *hit.detUnit(), tsos);
  return std::unique_ptr<SiPixelRecHit>(new SiPixelRecHit(std::get<0>(params), std::get<1>(params), std::get<2>(params), *hit.det(), hit.cluster()));
}

std::unique_ptr<SiStripRecHit2D> TkClonerImpl::operator()(SiStripRecHit2D const & hit, TrajectoryStateOnSurface const& tsos) const {
    /// FIXME: this only uses the first cluster and ignores the others
    const SiStripCluster&  clust = hit.stripCluster();  
    StripClusterParameterEstimator::LocalValues lv = 
      stripCPE->localParameters( clust, *hit.detUnit(), tsos);
    return std::unique_ptr<SiStripRecHit2D>{new SiStripRecHit2D(lv.first, lv.second, *hit.det(), hit.omniCluster())};
}

std::unique_ptr<SiStripRecHit1D> TkClonerImpl::operator()(SiStripRecHit1D const & hit, TrajectoryStateOnSurface const& tsos) const {
  /// FIXME: this only uses the first cluster and ignores the others
  const SiStripCluster&  clust = hit.stripCluster();  
  StripClusterParameterEstimator::LocalValues lv = 
    stripCPE->localParameters( clust, *hit.detUnit(), tsos);
  LocalError le(lv.second.xx(),0.,std::numeric_limits<float>::max()); //Correct??
  return std::unique_ptr<SiStripRecHit1D>{new SiStripRecHit1D(lv.first, le, *hit.det(), hit.omniCluster())};
}

std::unique_ptr<Phase2TrackerRecHit1D> TkClonerImpl::operator()(Phase2TrackerRecHit1D const & hit, TrajectoryStateOnSurface const& tsos) const {
  const Phase2TrackerCluster1D&  clust = hit.phase2OTCluster();
  const PixelGeomDetUnit & gdu = (const PixelGeomDetUnit &) *(hit.detUnit()) ;
  const PixelTopology * topo = &gdu.specificTopology();

  //FIXME:just temporary solution for phase2!
  float pitch_x = topo->pitch().first;
  float pitch_y = topo->pitch().second;
  float ix = clust.center();
  float iy = clust.column()+0.5; // halfway the column

  LocalPoint lp( topo->localX(ix), topo->localY(iy), 0 );          // x, y, z
  LocalError le( pow(pitch_x, 2) / 12, 0, pow(pitch_y, 2) / 12);   // e2_xx, e2_xy, e2_yy

  return std::unique_ptr<Phase2TrackerRecHit1D>{new Phase2TrackerRecHit1D(lp, le, *hit.det(), hit.cluster())};
}

TrackingRecHit::ConstRecHitPointer TkClonerImpl::makeShared(SiPixelRecHit const & hit, TrajectoryStateOnSurface const& tsos) const {
 // std::cout << "cloning " << typeid(hit).name() << std::endl;
  const SiPixelCluster& clust = *hit.cluster();  
  auto && params = pixelCPE->getParameters( clust, *hit.detUnit(), tsos);
  return std::make_shared<SiPixelRecHit>(std::get<0>(params), std::get<1>(params), std::get<2>(params), *hit.det(), hit.cluster());
}

TrackingRecHit::ConstRecHitPointer TkClonerImpl::makeShared(SiStripRecHit2D const & hit, TrajectoryStateOnSurface const& tsos) const {
  // std::cout << "cloning " << typeid(hit).name()	<< std::endl;
    /// FIXME: this only uses the first cluster and ignores the others
    const SiStripCluster&  clust = hit.stripCluster();  
    StripClusterParameterEstimator::LocalValues lv = 
      stripCPE->localParameters( clust, *hit.detUnit(), tsos);
    return std::make_shared<SiStripRecHit2D>(lv.first, lv.second, *hit.det(), hit.omniCluster());
}

TrackingRecHit::ConstRecHitPointer TkClonerImpl::makeShared(SiStripRecHit1D const & hit, TrajectoryStateOnSurface const& tsos) const {
  // std::cout << "cloning " << typeid(hit).name()	<< std::endl;
  /// FIXME: this only uses the first cluster and ignores the others
  const SiStripCluster&  clust = hit.stripCluster();  
  StripClusterParameterEstimator::LocalValues lv = 
    stripCPE->localParameters( clust, *hit.detUnit(), tsos);
  LocalError le(lv.second.xx(),0.,std::numeric_limits<float>::max()); //Correct??
  return  std::make_shared<SiStripRecHit1D>(lv.first, le, *hit.det(), hit.omniCluster());
}

TrackingRecHit::ConstRecHitPointer TkClonerImpl::makeShared(Phase2TrackerRecHit1D const & hit, TrajectoryStateOnSurface const& tsos) const {
  const Phase2TrackerCluster1D&  clust = hit.phase2OTCluster();
  const PixelGeomDetUnit & gdu = (const PixelGeomDetUnit &) *(hit.detUnit()) ;
  const PixelTopology * topo = &gdu.specificTopology();

  //FIXME:just temporary solution for phase2!
  float pitch_x = topo->pitch().first;
  float pitch_y = topo->pitch().second;
  float ix = clust.center();
  float iy = clust.column()+0.5; // halfway the column

  LocalPoint lp( topo->localX(ix), topo->localY(iy), 0 );          // x, y, z
  LocalError le( pow(pitch_x, 2) / 12, 0, pow(pitch_y, 2) / 12);   // e2_xx, e2_xy, e2_yy

  return std::make_shared<Phase2TrackerRecHit1D>( lp, le, *hit.det(), hit.cluster());
}


namespace {
#undef RecoTracker_TransientTrackingRecHit_TSiStripMatchedRecHit_RefitProj
#undef RecoTracker_TransientTrackingRecHit_TSiStripMatchedRecHit_RefitLGL
#ifdef RecoTracker_TransientTrackingRecHit_TSiStripMatchedRecHit_RefitLGL 
// Local lo Global lo Local
  inline LocalTrajectoryParameters gluedToStereo(const TrajectoryStateOnSurface &tsos, const GluedGeomDet *gdet) {    
    const BoundPlane &stripPlane = gdet->stereoDet()->surface();
    LocalPoint  lp = stripPlane.toLocal(tsos.globalPosition());
    LocalVector ld = stripPlane.toLocal(tsos.globalParameters().momentum());
    return LocalTrajectoryParameters(lp,ld,tsos.charge());
  }
#elif defined(RecoTracker_TransientTrackingRecHit_TSiStripMatchedRecHit_RefitProj)
  // A la RecHitProjector
  inline LocalTrajectoryParameters gluedToStereo(const TrajectoryStateOnSurface &tsos, const GluedGeomDet *gdet) {
    const BoundPlane &stripPlane = gdet->stereoDet()->surface();
    double delta = stripPlane.localZ( tsos.globalPosition());
    LocalVector ld = stripPlane.toLocal(tsos.globalParameters().momentum());
    LocalPoint  lp = stripPlane.toLocal(tsos.globalPosition()) - ld*delta/ld.z();
    return LocalTrajectoryParameters(lp,ld,tsos.charge());
  }
#else
  // Dummy
  inline const LocalTrajectoryParameters & gluedToStereo(const TrajectoryStateOnSurface &tsos, const GluedGeomDet *gdet) {
    return tsos.localParameters();
  }
#endif
}

std::unique_ptr<SiStripMatchedRecHit2D> TkClonerImpl::operator()(SiStripMatchedRecHit2D const & hit, TrajectoryStateOnSurface const& tsos) const {
    const GeomDet * det = hit.det();
    const GluedGeomDet *gdet = static_cast<const GluedGeomDet *> (det);
    LocalVector tkDir = (tsos.isValid() ? tsos.localDirection() : 
			 det->surface().toLocal( det->position()-GlobalPoint(0,0,0)));

    const SiStripCluster& monoclust   = hit.monoCluster();  
    const SiStripCluster& stereoclust = hit.stereoCluster();
    
    StripClusterParameterEstimator::LocalValues lvMono = 
      stripCPE->localParameters( monoclust, *gdet->monoDet(), tsos);
    StripClusterParameterEstimator::LocalValues lvStereo = 
      stripCPE->localParameters( stereoclust, *gdet->stereoDet(), gluedToStereo(tsos, gdet));
      
    SiStripRecHit2D monoHit = SiStripRecHit2D( lvMono.first, lvMono.second, 
					       *gdet->monoDet(),
					       hit.monoClusterRef());
    SiStripRecHit2D stereoHit = SiStripRecHit2D( lvStereo.first, lvStereo.second,
						 *gdet->stereoDet(),
						 hit.stereoClusterRef());
    
    // return theMatcher->match(&monoHit,&stereoHit,gdet,tkDir,true);
    std::unique_ptr<SiStripMatchedRecHit2D> temp = theMatcher->match(&monoHit,&stereoHit,gdet,tkDir,false);
    if(temp.get() == nullptr) {
      temp = std::unique_ptr<SiStripMatchedRecHit2D>(hit.clone());
    }
    return temp;
}

TrackingRecHit::ConstRecHitPointer TkClonerImpl::makeShared(SiStripMatchedRecHit2D const & hit, TrajectoryStateOnSurface const& tsos) const {
  ///std::cout << "cloning " << typeid(hit).name()	<< std::endl;
  return TrackingRecHit::ConstRecHitPointer((*this)(hit,tsos));
}

TrackingRecHit::ConstRecHitPointer TkClonerImpl::makeShared(ProjectedSiStripRecHit2D const & hit, TrajectoryStateOnSurface const& tsos) const {
  // std::cout << "cloning " << typeid(hit).name()	<< std::endl;
   return TrackingRecHit::ConstRecHitPointer((*this)(hit,tsos));
}

std::unique_ptr<ProjectedSiStripRecHit2D> TkClonerImpl::operator()(ProjectedSiStripRecHit2D const & hit, TrajectoryStateOnSurface const& tsos) const {
  const SiStripCluster& clust = hit.stripCluster();
  const GeomDetUnit * gdu = reinterpret_cast<const GeomDetUnit *>(hit.originalDet());
  //if (!gdu) std::cout<<"no luck dude"<<std::endl;
  StripClusterParameterEstimator::LocalValues lv = stripCPE->localParameters(clust, *gdu, tsos);

  // project...
  const GeomDet & det = *hit.det();
  const BoundPlane& gluedPlane = det.surface();
  const BoundPlane& hitPlane = gdu->surface();
  LocalVector tkDir = (tsos.isValid() ? tsos.localDirection() : 
		       det.surface().toLocal( det.position()-GlobalPoint(0,0,0)));

  auto delta = gluedPlane.localZ( hitPlane.position());
  LocalVector ldir = tkDir;
  LocalPoint lhitPos = gluedPlane.toLocal( hitPlane.toGlobal(lv.first));
  LocalPoint projectedHitPos = lhitPos - ldir * delta/ldir.z();                  

  LocalVector hitXAxis = gluedPlane.toLocal( hitPlane.toGlobal( LocalVector(1.f,0,0)));
  LocalError hitErr = lv.second;
  if (gluedPlane.normalVector().dot( hitPlane.normalVector()) < 0) {
    // the two planes are inverted, and the correlation element must change sign
    hitErr = LocalError( hitErr.xx(), -hitErr.xy(), hitErr.yy());
  }
  LocalError rotatedError = hitErr.rotate( hitXAxis.x(), hitXAxis.y());
  return std::unique_ptr<ProjectedSiStripRecHit2D>{new ProjectedSiStripRecHit2D(projectedHitPos, rotatedError, *hit.det(), *hit.originalDet(), hit.omniCluster())};
}


std::unique_ptr<ProjectedSiStripRecHit2D> TkClonerImpl::project(SiStripMatchedRecHit2D const & hit, bool mono, TrajectoryStateOnSurface const& tsos) const {
  const GeomDet & det = *hit.det();
  const GluedGeomDet & gdet = static_cast<const GluedGeomDet &> (det);
  const GeomDetUnit * odet = mono ? gdet.monoDet() : gdet.stereoDet();
  const BoundPlane& gluedPlane = det.surface();
  const BoundPlane& hitPlane = odet->surface();


  LocalVector tkDir = (tsos.isValid() ? tsos.localDirection() : 
		       det.surface().toLocal( det.position()-GlobalPoint(0,0,0)));

  const SiStripCluster& monoclust   = hit.monoCluster();  
  const SiStripCluster& stereoclust = hit.stereoCluster();

  StripClusterParameterEstimator::LocalValues lv;
  if (tsos.isValid()) 
    lv = mono ?
    stripCPE->localParameters( monoclust, *odet, tsos) :
    stripCPE->localParameters( stereoclust, *odet, gluedToStereo(tsos, &gdet));
  else 
    lv = stripCPE->localParameters( mono ? monoclust : stereoclust, *odet);


  auto delta = gluedPlane.localZ( hitPlane.position());
  LocalVector ldir = tkDir;
  LocalPoint lhitPos = gluedPlane.toLocal( hitPlane.toGlobal(lv.first));
  LocalPoint projectedHitPos = lhitPos - ldir * delta/ldir.z();                  

  LocalVector hitXAxis = gluedPlane.toLocal( hitPlane.toGlobal( LocalVector(1.f,0,0)));
  LocalError hitErr = lv.second;
  if (gluedPlane.normalVector().dot( hitPlane.normalVector()) < 0) {
    // the two planes are inverted, and the correlation element must change sign
    hitErr = LocalError( hitErr.xx(), -hitErr.xy(), hitErr.yy());
  }
  LocalError rotatedError = hitErr.rotate( hitXAxis.x(), hitXAxis.y());
  return std::unique_ptr<ProjectedSiStripRecHit2D>{
    new ProjectedSiStripRecHit2D(projectedHitPos, rotatedError, det, *odet, 
				 mono ? hit.monoClusterRef() : hit.stereoClusterRef() ) };
}

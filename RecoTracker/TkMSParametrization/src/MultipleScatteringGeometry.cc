#include "MultipleScatteringGeometry.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"

#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/GeomPropagators/interface/StraightLinePropagator.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "FWCore/Framework/interface/ESHandle.h"

using namespace GeomDetEnumerators;
using namespace std;

//------------------------------------------------------------------------------
const float MultipleScatteringGeometry::beamPipeR =  2.94;
const float MultipleScatteringGeometry::endflangesZ = 30;
const float MultipleScatteringGeometry::supportR = 19.;

//------------------------------------------------------------------------------
MultipleScatteringGeometry::MultipleScatteringGeometry(const edm::EventSetup &iSetup)
{

  edm::ESHandle<GeometricSearchTracker> track;
  iSetup.get<TrackerRecoGeometryRecord>().get( track ); 

  vector<BarrelDetLayer*> barrelLayers=track->barrelLayers();
  vector<BarrelDetLayer*>::const_iterator ib;
  vector<ForwardDetLayer*> forwardPosLayers=track->posForwardLayers();
  vector<ForwardDetLayer*> forwardNegLayers=track->negForwardLayers();
  vector<ForwardDetLayer*>::const_iterator ie;
  // barrelLayers = accessor.barrelLayers();
  for (ib = barrelLayers.begin(); ib != barrelLayers.end(); ib++)
    theLayers.push_back(*ib);

//   forwardLayers = accessor.negativeEndcapLayers();
  for (ie = forwardPosLayers.begin(); ie != forwardPosLayers.end(); ie++)
    theLayers.push_back(*ie);
  for (ie = forwardNegLayers.begin(); ie != forwardNegLayers.end(); ie++)
    theLayers.push_back(*ie);

}

//------------------------------------------------------------------------------
vector<MSLayer> MultipleScatteringGeometry::detLayers(const edm::EventSetup &iSetup) const
{
  vector<MSLayer> result;
  vector<const DetLayer*>::const_iterator il;
  for (il=theLayers.begin();il!=theLayers.end();il++) result.push_back(MSLayer(*il)); 
  return result;
}
    
//------------------------------------------------------------------------------
vector<MSLayer> MultipleScatteringGeometry::detLayers(float eta, float z,const edm::EventSetup &iSetup) const
{
  vector<MSLayer> result;
  GlobalPoint zero(0,0,z);
  float r=1; float dirZ = r*sinh(eta);
  GlobalVector dir(r,0.,dirZ);
  edm::ESHandle<MagneticField> pSetup;
  iSetup.get<IdealMagneticFieldRecord>().get(pSetup);
  FreeTrajectoryState fts( GlobalTrajectoryParameters(zero,dir,1,&(*pSetup)) );
  StraightLinePropagator propagator(&(*pSetup),alongMomentum);
  vector<const DetLayer*>::const_iterator il;
  TrajectoryStateOnSurface tsos;
  for (il = theLayers.begin(); il != theLayers.end(); il++) {
    bool contains=false;
//  if ((*il)->subDetector() != PixelBarrel && (*il)->subDetector()!= PixelEndcap) continue;

    if ( (*il)->location() == barrel ) {
      const BarrelDetLayer * bl = dynamic_cast<const BarrelDetLayer*>(*il);
      if (!bl) continue;
      tsos = propagator.propagate(fts, bl->specificSurface());
      if (!tsos.isValid()) continue;
      float r=bl->specificSurface().radius();
      float dr = bl->specificSurface().bounds().thickness();
      float z=bl->position().z();
      float dz = bl->specificSurface().bounds().length(); 
      PixelRecoRange<float> rRange(r-dr/2., r+dr/2.);
      PixelRecoRange<float> zRange(z-dz/2., z+dz/2.);
      contains =    rRange.inside(tsos.globalPosition().perp()) 
                 && zRange.inside(tsos.globalPosition().z());
      //     contains = bl->contains((*il)->toLocal(tsos.globalPosition()));
    }
    else if ((*il)->location() == endcap) {
      const ForwardDetLayer * fl = dynamic_cast<const ForwardDetLayer*>(*il);
      if (!fl) continue;
      if (fl->position().z() * eta < 0) continue;
      const BoundDisk & disk = fl->specificSurface();
      tsos = propagator.propagate(fts, disk);
      if (!tsos.isValid()) continue;
      float zMin = disk.position().z()-disk.bounds().thickness()/2;
      float zMax = disk.position().z()+disk.bounds().thickness()/2;
      PixelRecoRange<float> rRange(disk.innerRadius(), disk.outerRadius());
      PixelRecoRange<float> zRange(zMin, zMax);
      contains = rRange.inside(tsos.globalPosition().perp()) 
              && zRange.inside(tsos.globalPosition().z());
      //     contains = fl->contains(fl->toLocal(tsos.globalPosition()));
    }
    if (contains) result.push_back(MSLayer(*il));
  }
  return result;
}
//------------------------------------------------------------------------------
vector<MSLayer> MultipleScatteringGeometry::otherLayers(float eta,const edm::EventSetup &iSetup) const
{
  vector<MSLayer> result;
  // zero
  //  MSLayer zero(barrel, 0., MSLayer::Range(-15,15));
  //  result.push_back(zero);

  // beampipe
  MSLayer beampipe(barrel, beamPipeR, MSLayer::Range(-100,100));

  result.push_back(beampipe);

  // endflanges
  PixelRecoPointRZ endfPoint = (eta > 0) ?
      PixelRecoPointRZ(endflangesZ/sinh(eta), endflangesZ)
    : PixelRecoPointRZ(-endflangesZ/sinh(eta), -endflangesZ);
  if (0 < endfPoint.r() && endfPoint.r() < supportR) {
    MSLayer endflanges(endcap,endfPoint.z(),MSLayer::Range(0.1,supportR-0.1));
    result.push_back(endflanges);
  }

  // support
  MSLayer support( barrel,supportR, MSLayer::Range(-280,280));
  result.push_back(support);

  return result;
}

  


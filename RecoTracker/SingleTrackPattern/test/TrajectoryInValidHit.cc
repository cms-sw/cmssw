#include "RecoTracker/SingleTrackPattern/test/TrajectoryInValidHit.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateCombiner.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementError.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementVector.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "Geometry/TrackerGeometryBuilder/interface/GluedGeomDet.h"


using namespace std;
TrajectoryInValidHit::TrajectoryInValidHit( const TrajectoryMeasurement& tm, const TrackerGeometry* tracker)
{
  theCombinedPredictedState = TrajectoryStateCombiner().combine( tm.forwardPredictedState(),
  								 tm.backwardPredictedState());

  theHit = tm.recHit();  
  
  unsigned int iidd = theHit->geographicalId().rawId();
  TrajectoryStateTransform tsostransform;
  PTrajectoryStateOnDet* combinedptsod=tsostransform.persistentState( theCombinedPredictedState,iidd);
  
  
  StripSubdetector strip=StripSubdetector(iidd);
  unsigned int subid=strip.subdetId();
  unsigned int laytib = 1000;
  unsigned int laytob = 1000;
  float xB = 0.; 
  float yB = 0.;
  if (subid ==  StripSubdetector::TIB) { 
    TIBDetId tibid(iidd);
    laytib =tibid.layer();
    xB = 0.3;
    yB = 0.5;
  }
  if (subid ==  StripSubdetector::TOB) { 
    TOBDetId tobid(iidd);
    laytob =tobid.layer();
    xB = 0.3;
    yB = 1.0;
  }
  

  
  LocalVector monoco, stereoco;
  LocalPoint pmonoco, pstereoco;

  const GeomDetUnit * monodet;
  const GeomDetUnit * stereodet;
 
  if (laytib == 1 || laytib == 2 || laytob == 1 || laytob == 2){
    
    GluedGeomDet * gdet=(GluedGeomDet *)tracker->idToDet(theHit->geographicalId());    
    GlobalVector gtrkdirco=gdet->toGlobal(combinedptsod->parameters().momentum());

    monodet=gdet->monoDet();
    monoco=monodet->toLocal(gtrkdirco);
    pmonoco=project(gdet,monodet,combinedptsod->parameters().position(),monoco);
    
 
    RPhilocX_temp = pmonoco.x(); 
    RPhilocY_temp = pmonoco.y(); 
     
    stereodet = gdet->stereoDet();
    stereoco=stereodet->toLocal(gtrkdirco);
    pstereoco=project(gdet,stereodet,combinedptsod->parameters().position(),stereoco);
    
    StereolocX_temp = pstereoco.x(); 
    StereolocY_temp = pstereoco.y(); 
  }
  else {
    RPhilocX_temp = theCombinedPredictedState.localPosition().x();
    RPhilocY_temp = theCombinedPredictedState.localPosition().y();
    StereolocX_temp = 1000.;
    StereolocY_temp = 1000.;
    monodet = (GeomDetUnit*)theHit->det();
    stereodet = (GeomDetUnit*)theHit->det();

  }


  // Restrict the bound regions for better understanding of the modul assignment. 

  LocalPoint BoundedPointRphi;
  LocalPoint BoundedPointSte;
  float xRphi,yRphi,zz;
  float xSte,ySte;

  // Insert the bounded values 

  if (RPhilocX_temp < 0. ) xRphi  = RPhilocX_temp - xB;
  else  xRphi = RPhilocX_temp + xB;
  if (RPhilocY_temp < 0. ) yRphi = RPhilocY_temp - yB;
  else  yRphi = RPhilocY_temp + yB;
    
  if (StereolocX_temp < 0. ) xSte = StereolocX_temp - xB;
  else  xSte = StereolocX_temp + xB;
  if (StereolocY_temp < 0. ) ySte = StereolocY_temp - yB;
  else  ySte = StereolocY_temp + yB;


  zz = theCombinedPredictedState.localPosition().z();
  
  
  BoundedPointRphi = LocalPoint(xRphi,yRphi,zz);
  BoundedPointSte = LocalPoint(xSte,ySte,zz);
  
  
  // ---> RPhi Stereo modules 
  if ( monodet->surface().bounds().inside(BoundedPointRphi)) {
    
    RPhilocX = RPhilocX_temp;
    RPhilocY = RPhilocY_temp;
  }
  else {
    RPhilocX = 2000.;
    RPhilocY = 2000.;
  }
  // ---> TIB Stereo modules 
  if ( stereodet->surface().bounds().inside(BoundedPointSte)) {
    StereolocX = StereolocX_temp;
    StereolocY = StereolocY_temp;
  }
  else {
    StereolocX = 2000.;
    StereolocY = 2000.;
  }
}

double TrajectoryInValidHit::localRPhiX() const
{
  return RPhilocX;
}
double TrajectoryInValidHit::localRPhiY() const
{
  return RPhilocY;
}
double TrajectoryInValidHit::localStereoX() const
{
  return StereolocX;
}
double TrajectoryInValidHit::localStereoY() const
{
  return StereolocY;
}
double TrajectoryInValidHit::localZ() const
{
  return theCombinedPredictedState.localPosition().z();
}
double TrajectoryInValidHit::localErrorX() const
{
  return sqrt(theCombinedPredictedState.localError().positionError().xx());
}
double TrajectoryInValidHit::localErrorY() const
{
  return sqrt(theCombinedPredictedState.localError().positionError().yy());
}
double TrajectoryInValidHit::globalX() const
{
  return theCombinedPredictedState.globalPosition().x();
}

double TrajectoryInValidHit::globalY() const
{
  return theCombinedPredictedState.globalPosition().y();
}
double TrajectoryInValidHit::globalZ() const
{
  return theCombinedPredictedState.globalPosition().z();
}


bool TrajectoryInValidHit::InValid() const
{
  return IsInvHit;
}

LocalPoint TrajectoryInValidHit::project(const GeomDet *det,const GeomDet* projdet,LocalPoint position,LocalVector trackdirection)const
{
  
  GlobalPoint globalpoint=(det->surface()).toGlobal(position);
  
  // position of the initial and final point of the strip in glued local coordinates
  LocalPoint projposition=(projdet->surface()).toLocal(globalpoint);
  
  //correct the position with the track direction
  
  float scale=-projposition.z()/trackdirection.z();
  
  projposition+= scale*trackdirection;
  
  return projposition;
}

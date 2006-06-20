#include "RecoTracker/TkDetLayers/interface/BladeShapeBuilderFromDet.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TrackingTools/DetLayers/interface/PhiLess.h"
#include "Geometry/Surface/interface/BoundingBox.h"

#include <iomanip>

using namespace std;

BoundDiskSector* 
BladeShapeBuilderFromDet::operator()( const vector<const GeomDet*>& dets) const
{
  // find mean position
  typedef Surface::PositionType::BasicVectorType Vector;
  Vector posSum(0,0,0);
  for (vector<const GeomDet*>::const_iterator i=dets.begin(); i!=dets.end(); i++) {
    posSum += (**i).surface().position().basicVector();
  }
  Surface::PositionType meanPos( 0.,0.,posSum.z()/float(dets.size()) );

  // temporary plane - for the computation of bounds
  Surface::RotationType rotation = computeRotation( dets, meanPos);
  BoundPlane tmpPlane( meanPos, rotation);


  pair<DiskSectorBounds,GlobalVector> bo = 
    computeBounds( dets,tmpPlane );
  GlobalPoint pos = meanPos+bo.second;
  //edm::LogInfo(TkDetLayers) << "global pos in operator: " << pos ;
  return new BoundDiskSector( pos, rotation, bo.first);
}

pair<DiskSectorBounds, GlobalVector>
BladeShapeBuilderFromDet::computeBounds( const vector<const GeomDet*>& dets,
					 const BoundPlane& plane) const
{
  Surface::PositionType tmpPos = dets.front()->surface().position();


  float rmin(plane.toLocal(tmpPos).perp());
  float rmax(plane.toLocal(tmpPos).perp());
  float zmin(plane.toLocal(tmpPos).z());
  float zmax(plane.toLocal(tmpPos).z());
  float phimin(plane.toLocal(tmpPos).phi());
  float phimax(plane.toLocal(tmpPos).phi());

  for(vector<const GeomDet*>::const_iterator it=dets.begin(); it!=dets.end(); it++){
    vector<GlobalPoint> corners = BoundingBox().corners( (*it)->specificSurface() );

    for(vector<GlobalPoint>::const_iterator i=corners.begin();
	i!=corners.end(); i++) {

      float r   = plane.toLocal(*i).perp();
      float z   = plane.toLocal(*i).z();
      float phi = plane.toLocal(*i).phi();
      rmin = min( rmin, r);
      rmax = max( rmax, r);
      zmin = min( zmin, z);
      zmax = max( zmax, z);
      if ( PhiLess()( phi, phimin)) phimin = phi;
      if ( PhiLess()( phimax, phi)) phimax = phi;
    }
  }

  if (!PhiLess()(phimin, phimax)) 
    edm::LogError("TkDetLayers") << " ForwardDiskSectorBuilderFromDet : " 
				 << "Something went wrong with Phi Sorting !" ;
  float zPos = (zmax+zmin)/2.;
  float phiWin = phimax - phimin;
  float phiPos = (phimax+phimin)/2.;
  float rmed = (rmin+rmax)/2.;
  if ( phiWin < 0. ) {
    if ( (phimin < Geom::pi() / 2.) || (phimax > -Geom::pi()/2.) ){
      edm::LogError("TkDetLayers") << " something strange going on, please check " ;
    }
    //edm::LogInfo(TkDetLayers) << " Wedge at pi: phi " << phimin << " " << phimax << " " << phiWin 
    //	 << " " << 2.*Geom::pi()+phiWin << " " ;
    phiWin += 2.*Geom::pi();
    phiPos += Geom::pi(); 
  }
  
  LocalVector localPos( rmed*cos(phiPos), rmed*sin(phiPos), zPos);

  LogDebug("TkDetLayers") << "localPos in computeBounds: " << localPos << "\n"
			  << "rmin:   " << rmin << "\n"
			  << "rmax:   " << rmax << "\n"
			  << "zmin:   " << zmin << "\n"
			  << "zmax:   " << zmax << "\n"
			  << "phiWin: " << phiWin ;

  return make_pair(DiskSectorBounds(rmin,rmax,zmin,zmax,phiWin),
		   plane.toGlobal(localPos) );

}


Surface::RotationType 
BladeShapeBuilderFromDet::computeRotation( const vector<const GeomDet*>& dets,
					   const Surface::PositionType& meanPos) const {
  const BoundPlane& plane = dets.front()->surface();
  
  GlobalVector xAxis;
  GlobalVector yAxis;
  
  GlobalVector planeXAxis = plane.toGlobal( LocalVector( 1, 0, 0));
  yAxis = planeXAxis;

  /*
  if ( planeXAxis.x() * meanPos.x() + planeXAxis.y() * meanPos.y() > 0) {
    yAxis = planeXAxis;
  }
  else {
    edm::LogInfo(TkDetLayers) << "something weird in BladeShapeBuilderFromDet::computeRotation." 
	 << "planeXAxis points inward.." ;
    yAxis =  -planeXAxis;
  }
  */

  GlobalVector planeYAxis = plane.toGlobal( LocalVector( 0, 1, 0));
  GlobalVector n = planeYAxis.cross( yAxis);
  
  if (n.z() > 0) {
    xAxis = planeYAxis;
  }
  else {
    xAxis = -planeYAxis;
  }
  
  //   edm::LogInfo(TkDetLayers) << "Creating rotation with x,y axis " 
  //        << xAxis << ", " << yAxis ;
  return Surface::RotationType( xAxis, yAxis);
}




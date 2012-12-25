#include "BladeShapeBuilderFromDet.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TrackingTools/DetLayers/interface/PhiLess.h"
#include "DataFormats/GeometrySurface/interface/BoundingBox.h"

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
  Plane tmpPlane( meanPos, rotation);


  auto bo = 
    computeBounds( dets,tmpPlane );
  GlobalPoint pos = meanPos+bo.second;
  //edm::LogInfo(TkDetLayers) << "global pos in operator: " << pos ;
  return new BoundDiskSector( pos, rotation, bo.first);
}

pair<DiskSectorBounds *, GlobalVector>
BladeShapeBuilderFromDet::computeBounds( const vector<const GeomDet*>& dets,
					 const Plane& plane) const
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
    // in addition to the corners we have to check the middle of the 
    // det +/- length/2, since the min (max) radius for typical fw
    // dets is reached there
        
    float rdet = (*it)->position().perp();
    float height  = (*it)->surface().bounds().width();
    rmin = min( rmin, rdet-height/2.F);
    rmax = max( rmax, rdet+height/2.F);  
    

  }

  if (!PhiLess()(phimin, phimax)) 
    edm::LogError("TkDetLayers") << " BladeShapeBuilderFromDet : " 
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

  return make_pair(new DiskSectorBounds(rmin,rmax,zmin,zmax,phiWin),
		   plane.toGlobal(localPos) );

}


Surface::RotationType 
BladeShapeBuilderFromDet::computeRotation( const vector<const GeomDet*>& dets,
					   const Surface::PositionType& meanPos) const
{
  const Plane& plane = dets.front()->surface();
  
  GlobalVector xAxis;
  GlobalVector yAxis;
  GlobalVector zAxis;
  
  GlobalVector planeXAxis    = plane.toGlobal( LocalVector( 1, 0, 0));
  GlobalPoint  planePosition = plane.position();

  if(planePosition.x()*planeXAxis.x()+planePosition.y()*planeXAxis.y() > 0.){
    yAxis = planeXAxis;
  }else{
    yAxis = -planeXAxis;
  }

  GlobalVector planeZAxis = plane.toGlobal( LocalVector( 0, 0, 1));
  if(planeZAxis.z()*planePosition.z() > 0.){
    zAxis = planeZAxis;
  }else{
    zAxis = -planeZAxis;
  }

  xAxis = yAxis.cross( zAxis);
  
  return Surface::RotationType( xAxis, yAxis);
}




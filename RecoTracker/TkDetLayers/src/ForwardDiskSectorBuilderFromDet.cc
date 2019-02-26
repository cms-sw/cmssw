#include "ForwardDiskSectorBuilderFromDet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/GeometrySurface/interface/TrapezoidalPlaneBounds.h"
#include "DataFormats/GeometryVector/interface/VectorUtil.h"
#include "TrackingTools/DetLayers/interface/DetLayerException.h"

using namespace std;

// Warning, remember to assign this pointer to a ReferenceCountingPointer!
BoundDiskSector* 
ForwardDiskSectorBuilderFromDet::operator()( const vector<const GeomDet*>& dets) const
{
  // check that the dets are all at about the same radius and z 
  float rcheck = dets.front()->surface().position().perp();
  float zcheck = dets.front()->surface().position().z();
  for ( vector<const GeomDet*>::const_iterator i = dets.begin(); i != dets.end(); i++){
    float rdiff = (**i).surface().position().perp()-rcheck;
    if ( std::abs(rdiff) > 1.) 
      edm::LogError("TkDetLayers") << " ForwardDiskSectorBuilderFromDet: Trying to build Petal Wedge from " 
				   << "Dets at different radii !! Delta_r = " << rdiff ;
    float zdiff = zcheck - (**i).surface().position().z();
    if ( std::abs(zdiff) > 0.8) 
      edm::LogError("TkDetLayers") << " ForwardDiskSectorBuilderFromDet: Trying to build Petal Wedge from " 
				   << "Dets at different z positions !! Delta_z = " << zdiff ;
  }

  auto bo = computeBounds( dets );

  Surface::PositionType pos( bo.second.x(), bo.second.y(), bo.second.z() );
  Surface::RotationType rot = computeRotation( dets, pos);
  return new BoundDiskSector( pos, rot, bo.first);
}

pair<DiskSectorBounds*, GlobalVector>
ForwardDiskSectorBuilderFromDet::computeBounds( const vector<const GeomDet*>& dets) const
{
  // go over all corners and compute maximum deviations 
  float rmin = (**(dets.begin())).surface().position().perp();
  float rmax(rmin); 
  float zmin((**(dets.begin())).surface().position().z());
  float zmax(zmin);
  float phimin((**(dets.begin())).surface().position().phi());
  float phimax(phimin);


  for (vector<const GeomDet*>::const_iterator idet=dets.begin();
       idet != dets.end(); idet++) {
    vector<const GeomDet*> detUnits = (**idet).components();
    if( !detUnits.empty() ){
      for (vector<const GeomDet*>::const_iterator detu=detUnits.begin();
	   detu!=detUnits.end(); detu++) {
	// edm::LogInfo(TkDetLayers) << " Builder: Position of detUnit :"<< (**detu).position() ;      
	vector<GlobalPoint> corners = computeTrapezoidalCorners(*detu) ;
	for (vector<GlobalPoint>::const_iterator i=corners.begin();
	     i!=corners.end(); i++) {
	  float r = i->perp();
	  float z = i->z();
	  float phi = i->phi();
	  rmin = min( rmin, r);
	  rmax = max( rmax, r);
	  zmin = min( zmin, z);
	  zmax = max( zmax, z);
	  if ( Geom::phiLess( phi, phimin)) phimin = phi;
	  if ( Geom::phiLess( phimax, phi)) phimax = phi;
	}
	// in addition to the corners we have to check the middle of the 
	// det +/- length/2, since the min (max) radius for typical fw
	// dets is reached there
	float rdet  = (**detu).position().perp();
	float len   = (**detu).surface().bounds().length();
	float width = (**detu).surface().bounds().width();
	
	GlobalVector xAxis = (**detu).toGlobal(LocalVector(1,0,0));
	GlobalVector yAxis = (**detu).toGlobal(LocalVector(0,1,0));
	GlobalVector perpDir = GlobalVector( (**detu).position() - GlobalPoint(0,0,(**detu).position().z()) );
	
	double xAxisCos = xAxis.unit().dot(perpDir.unit());
	double yAxisCos = yAxis.unit().dot(perpDir.unit());
	
	if( std::abs(xAxisCos) > std::abs(yAxisCos) ) {
	  rmin = min( rmin, rdet-width/2.F);
	  rmax = max( rmax, rdet+width/2.F);
	}else{
	  rmin = min( rmin, rdet-len/2.F);
	  rmax = max( rmax, rdet+len/2.F);
	}     
      }
    }else{
      vector<GlobalPoint> corners = computeTrapezoidalCorners(*idet) ;
      for (vector<GlobalPoint>::const_iterator i=corners.begin();
	   i!=corners.end(); i++) {
	float r = i->perp();
	float z = i->z();
	float phi = i->phi();
	rmin = min( rmin, r);
	rmax = max( rmax, r);
	zmin = min( zmin, z);
	zmax = max( zmax, z);
	if ( Geom::phiLess( phi, phimin)) phimin = phi;
	if ( Geom::phiLess( phimax, phi)) phimax = phi;
      }
      // in addition to the corners we have to check the middle of the 
      // det +/- length/2, since the min (max) radius for typical fw
      // dets is reached there

      float rdet  = (**idet).position().perp();
      float len   = (**idet).surface().bounds().length();
      float width = (**idet).surface().bounds().width();

      GlobalVector xAxis = (**idet).toGlobal(LocalVector(1,0,0));
      GlobalVector yAxis = (**idet).toGlobal(LocalVector(0,1,0));
      GlobalVector perpDir = GlobalVector( (**idet).position() - GlobalPoint(0,0,(**idet).position().z()) );

      double xAxisCos = xAxis.unit().dot(perpDir.unit());
      double yAxisCos = yAxis.unit().dot(perpDir.unit());
      
      if( std::abs(xAxisCos) > std::abs(yAxisCos) ) {
	rmin = min( rmin, rdet-width/2.F);
	rmax = max( rmax, rdet+width/2.F);
      }else{
	rmin = min( rmin, rdet-len/2.F);
	rmax = max( rmax, rdet+len/2.F);
      }     
    }
  }
  
  
  if (!Geom::phiLess(phimin, phimax)) 
    edm::LogError("TkDetLayers") << " ForwardDiskSectorBuilderFromDet : " 
				 << "Something went wrong with Phi Sorting !" ;
  float zPos = (zmax+zmin)/2.;
  float phiWin = phimax - phimin;
  float phiPos = (phimax+phimin)/2.;
  float rmed = (rmin+rmax)/2.;
  if ( phiWin < 0. ) {
    if ( (phimin < Geom::pi() / 2.) || (phimax > -Geom::pi()/2.) ){
      edm::LogError("TkDetLayers") << " Debug: something strange going on, please check " ;
    }
    // edm::LogInfo(TkDetLayers) << " Wedge at pi: phi " << phimin << " " << phimax << " " << phiWin 
    //	 << " " << 2.*Geom::pi()+phiWin << " " ;
    phiWin += 2.*Geom::pi();
    phiPos += Geom::pi(); 
  }
  GlobalVector pos( rmed*cos(phiPos), rmed*sin(phiPos), zPos);
  return make_pair(new DiskSectorBounds(rmin,rmax,zmin-zPos,zmax-zPos,phiWin), pos);
}

Surface::RotationType 
ForwardDiskSectorBuilderFromDet::computeRotation( const vector<const GeomDet*>& dets,
						  Surface::PositionType pos) const {
  
  GlobalVector yAxis = ( GlobalVector( pos.x(), pos.y(), 0.)).unit();
  
  GlobalVector zAxis( 0., 0., 1.);
  GlobalVector xAxis = yAxis.cross( zAxis);
  
  return Surface::RotationType( xAxis, yAxis);
}


vector<GlobalPoint> 
ForwardDiskSectorBuilderFromDet::computeTrapezoidalCorners( const GeomDet* det) const {


  const Plane& plane( det->specificSurface());
  
  const TrapezoidalPlaneBounds* myBounds( static_cast<const TrapezoidalPlaneBounds*>(&(plane.bounds())));
  
  /*
  if (myBounds == 0) {
    string errmsg="ForwardDiskSectorBuilderFromDet: problems with dynamic cast to trapezoidal bounds for DetUnits";
    throw DetLayerException(errmsg);
    edm::LogError("TkDetLayers") << errmsg ;
  }
  */

  // array<const float, 4> 
  auto const & parameters = (*myBounds).parameters();

  if ( parameters[0] == 0 ) {
    edm::LogError("TkDetLayers") << "ForwardDiskSectorBuilder: something weird going on !" ;
    edm::LogError("TkDetLayers") << " Trapezoidal parameters of GeomDet (L2/L1/T/H): " ;
    for (int i = 0; i < 4; i++ )     edm::LogError("TkDetLayers") << "  " 
								  << 2.*parameters[i] 
								  << "\n";  
  }


  float hbotedge = parameters[0];
  float htopedge = parameters[1];
  float hapothem = parameters[3];
  float hthick   = parameters[2];

  vector<GlobalPoint> corners;

  corners.push_back( plane.toGlobal( LocalPoint( -htopedge, hapothem, hthick)));
  corners.push_back( plane.toGlobal( LocalPoint( -htopedge, hapothem, -hthick)));
  corners.push_back( plane.toGlobal( LocalPoint(  htopedge, hapothem, hthick)));
  corners.push_back( plane.toGlobal( LocalPoint(  htopedge, hapothem, -hthick)));
  corners.push_back( plane.toGlobal( LocalPoint(  hbotedge, -hapothem, hthick)));
  corners.push_back( plane.toGlobal( LocalPoint(  hbotedge, -hapothem, -hthick)));
  corners.push_back( plane.toGlobal( LocalPoint( -hbotedge, -hapothem, hthick)));
  corners.push_back( plane.toGlobal( LocalPoint( -hbotedge, -hapothem, -hthick)));

  return corners;
}

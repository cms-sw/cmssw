#include "TrackingTools/DetLayers/interface/ForwardRingDiskBuilderFromDet.h"
#include "Geometry/Surface/interface/SimpleDiskBounds.h"
#include "Geometry/CommonDetAlgo/interface/BoundingBox.h"


// Warning, remember to assign this pointer to a ReferenceCountingPointer!
BoundDisk* 
ForwardRingDiskBuilderFromDet::operator()( const vector<const GeomDet*>& dets) const
{
  pair<SimpleDiskBounds,float> bo = 
    computeBounds( dets );

//   cout << "Creating disk at Z: " << bo.second << endl
//        << "Bounds are (rmin/rmax/thick) " << bo.first.innerRadius()
//        << " / " <<  bo.first.outerRadius()
//        << " / " <<  bo.first.thickness()  << endl;

//   typedef Det::PositionType::BasicVectorType Vector; 
//   Vector posSum(0,0,0);
//   for (vector<Det*>::const_iterator i=dets.begin(); i!=dets.end(); i++) {
//     Vector pp = (**i).position().basicVector();
//     //    cout << "  "<< (int) ( i-dets.begin()) << " at " << pp << endl;
//     posSum += pp;
//   }
//   Det::PositionType meanPos( posSum/float(dets.size()));
//   cout << "  meanPos "<< meanPos <<endl;

  Surface::PositionType pos(0.,0.,bo.second);
  Surface::RotationType rot;
  return new BoundDisk( pos, rot, bo.first);
}

pair<SimpleDiskBounds, float>
ForwardRingDiskBuilderFromDet::computeBounds( const vector<const GeomDet*>& dets) const
{
  // go over all corners and compute maximum deviations from mean pos.
  float rmin((**(dets.begin())).surface().position().perp());
  float rmax(rmin); 
  float zmin((**(dets.begin())).surface().position().z());
  float zmax(zmin);
  for (vector<const GeomDet*>::const_iterator idet=dets.begin();
       idet != dets.end(); idet++) {
    
    /* ---- original implementation. Is it obsolete?
    vector<DetUnit*> detUnits = (**idet).detUnits();
    for (vector<DetUnit*>::const_iterator detu=detUnits.begin();
    detu!=detUnits.end(); detu++) {
    vector<GlobalPoint> corners = BoundingBox().corners(
    dynamic_cast<const BoundPlane&>((**detu).surface()));
    }
    ----- */
    vector<GlobalPoint> corners = BoundingBox().corners( (**idet).specificSurface() );
    for (vector<GlobalPoint>::const_iterator i=corners.begin();
	 i!=corners.end(); i++) {
      float r = i->perp();
      float z = i->z();
      rmin = min( rmin, r);
      rmax = max( rmax, r);
      zmin = min( zmin, z);
      zmax = max( zmax, z);
    }
    // in addition to the corners we have to check the middle of the 
    // det +/- length/2, since the min (max) radius for typical fw
    // dets is reached there
    float rdet = (**idet).surface().position().perp();
    float len = (**idet).surface().bounds().length();
    rmin = min( rmin, rdet-len/2.F);
    rmax = max( rmax, rdet+len/2.F);      
  }

  
  float zPos = (zmax+zmin)/2.;
  return make_pair(SimpleDiskBounds(rmin,rmax,zmin-zPos,zmax-zPos), zPos);
}

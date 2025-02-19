#ifndef TkDetLayers_TECPetal_h
#define TkDetLayers_TECPetal_h


#include "TrackingTools/DetLayers/interface/GeometricSearchDet.h"
#include "TECWedge.h"
#include "BoundDiskSector.h"


/** A concrete implementation for TEC petals
 */

#pragma GCC visibility push(hidden)
class TECPetal : public GeometricSearchDetWithGroups {
 public:  
  // GeometricSearchDet interface  
  virtual const BoundSurface& surface() const{return *theDiskSector;}


  //Extension of the interface
  virtual const BoundDiskSector& specificSurface() const {return *theDiskSector;}

 protected:
  // it needs to be initialized somehow ins the derived class
  ReferenceCountingPointer<BoundDiskSector> theDiskSector;
  
};


#pragma GCC visibility pop
#endif 

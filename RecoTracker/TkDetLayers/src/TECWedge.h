#ifndef TkDetLayers_TECWedge_h
#define TkDetLayers_TECWedge_h

#include "TrackingTools/DetLayers/interface/GeometricSearchDet.h"
#include "DataFormats/GeometrySurface/interface/BoundDiskSector.h"

/** A concrete implementation for TEC layer 
 *  built out of TECPetals
 */

#pragma GCC visibility push(hidden)
class TECWedge : public GeometricSearchDet {
public:
  TECWedge() : GeometricSearchDet(true) {}

  // GeometricSearchDet interface
  const BoundSurface& surface() const final { return *theDiskSector; }

  //Extension of the interface
  virtual const BoundDiskSector& specificSurface() const final { return *theDiskSector; }

protected:
  // it needs to be initialized somehow ins the derived class
  ReferenceCountingPointer<BoundDiskSector> theDiskSector;
};

#pragma GCC visibility pop
#endif

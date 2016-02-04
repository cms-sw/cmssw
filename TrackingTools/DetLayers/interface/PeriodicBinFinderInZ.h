#ifndef DetLayers_PeriodicBinFinderInZ_H
#define DetLayers_PeriodicBinFinderInZ_H

#include "Utilities/BinningTools/interface/BaseBinFinder.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include <cmath>

/** Bin finder along the global Z for (almost) equidistant bins.
 *  The bins are computed from GeomDet positions.
 */

template <class T>
class PeriodicBinFinderInZ : public BaseBinFinder<T> {
public:

  PeriodicBinFinderInZ() : theNbins(0), theZStep(0), theZOffset(0) {}

  PeriodicBinFinderInZ(std::vector<const GeomDet*>::const_iterator first,
		       std::vector<const GeomDet*>::const_iterator last) :
    theNbins( last-first) 
  {
    float zFirst = (**first).surface().position().z();
    theZStep = ((**(last-1)).surface().position().z() - zFirst) / (theNbins-1);
    theZOffset = zFirst - 0.5*theZStep;
  }

  /// returns an index in the valid range for the bin that contains Z
  virtual int binIndex( T z) const {
    return binIndex( int((z-theZOffset)/theZStep));
  }

  /// returns an index in the valid range
  virtual int binIndex( int i) const {
    return std::min( std::max( i, 0), theNbins-1);
  }
   
  /// the middle of the bin 
  virtual T binPosition( int ind) const {
    return theZOffset + theZStep * ( ind + 0.5);
  }

private:

  int theNbins;
  T theZStep;
  T theZOffset;

};
#endif

#ifndef GeneralBinFinderInZ_H
#define GeneralBinFinderInZ_H

/** \class GeneralBinFinderInZ
 * A Z bin finder for a non-periodic group of detectors.
 */

#include "TrackingTools/DetLayers/interface/BaseBinFinder.h"
#include "TrackingTools/DetLayers/interface/GeometricSearchDet.h"
#include <cmath>

using namespace std;

template <class T>
class GeneralBinFinderInZforGeometricSearchDet : public BaseBinFinder<T> {
public:

  GeneralBinFinderInZforGeometricSearchDet() : theNbins(0), theZStep(0), theZOffset(0) {}

  GeneralBinFinderInZforGeometricSearchDet(
	              vector<const GeometricSearchDet*>::const_iterator first,
		      vector<const GeometricSearchDet*>::const_iterator last) :
    theNbins( last-first)
  {
    theBins.reserve(theNbins);
    for (vector<const GeometricSearchDet*>::const_iterator i=first; i<last-1; i++) {
      theBins.push_back((**i).position().z());
      theBorders.push_back(((**i).position().z() + 
			    (**(i+1)).position().z()) / 2.);
    }

    theZOffset = theBorders.front(); 
    theZStep = (theBorders.back() - theBorders.front()) / (theNbins-2);
  }

  /// returns an index in the valid range for the bin closest to Z
  virtual int binIndex( T z) const {
    int bin = binIndex( int((z-theZOffset)/theZStep)+1);

    // check left border
    if (bin > 0) {
      if ( z < theBorders[bin-1]) {
	// z is to the left of the left border, the correct bin is left
	for (int i=bin-1; ; i--) {
	  if (i <= 0) return 0;  
	  if ( z > theBorders[i-1]) return i;
	}
      }
    } 
    else return 0;

    // check right border
    if (bin < theNbins-1) {
      if ( z > theBorders[bin]) {
	// z is to the right of the right border, the correct bin is right
	for (int i=bin+1; ; i++) {
	  if (i >= theNbins-1) return theNbins-1;  
	  if ( z < theBorders[i]) return i;
	}
      }
    }
    else return theNbins-1;

    // if we arrive here it means that the bin is ok 
    return bin;
  }

  /// returns an index in the valid range
  virtual int binIndex( int i) const {
    return min( max( i, 0), theNbins-1);
  }
   
  /// the middle of the bin.
  virtual T binPosition( int ind) const {
    return theBins[binIndex(ind)];
  }

  static double pi() { return 3.141592653589793238;}
  static double twoPi() { return 2.*pi();}

private:

  int theNbins;
  T theZStep;
  T theZOffset;
  vector<float> theBorders;
  vector<T> theBins;
};
#endif

#ifndef BinningTools_GenericBinFinderInZ_H
#define BinningTools_GenericBinFinderInZ_H

/** \class GenericBinFinderInZ
 * A Z bin finder for a non-periodic group of detectors.
 * The template argument G should be something
 * with a surface() method, such as a GeomDet
 * or a GeometricSearchDet
 */

#include "Utilities/BinningTools/interface/BaseBinFinder.h"
#include <cmath>
#include <vector>

template <class T, class G>
class GenericBinFinderInZ : public BaseBinFinder<T> {
public:
  typedef typename std::vector<const G*>::const_iterator ConstItr;
  GenericBinFinderInZ() : theNbins(0), theZStep(0), theZOffset(0) {}

  GenericBinFinderInZ(ConstItr first, ConstItr last) :
    theNbins( last-first)
  {
    theBins.reserve(theNbins);
    theBorders.reserve(theNbins-1);
    for (ConstItr i=first; i<last-1; i++) {
      theBins.push_back((**i).position().z());
      theBorders.push_back(((**i).position().z() + 
			    (**(i+1)).position().z()) / 2.);
    }
    theBins.push_back((**(last-1)).position().z());

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
    return std::min( std::max( i, 0), theNbins-1);
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
  std::vector<float> theBorders;
  std::vector<T> theBins;
};
#endif

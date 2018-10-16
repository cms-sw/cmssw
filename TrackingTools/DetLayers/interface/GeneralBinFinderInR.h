#ifndef TrackingTools_DetLayers_GeneralBinFinderInR_H
#define TrackingTools_DetLayers_GeneralBinFinderInR_H

/** \class GeneralBinFinderInR
 *  A R binfinder for a non-periodic group of detectors.
 *
 *  \author N. Amapane - INFN Torino
 */

#include <Utilities/BinningTools/interface/BaseBinFinder.h>
#include "TrackingTools/DetLayers/interface/RBorderFinder.h"

#include <cmath>
#include <vector>

template <class T>
class GeneralBinFinderInR : public BaseBinFinder<T>{
public:
  
  typedef RBorderFinder::Det Det; //FIXME!!!

  GeneralBinFinderInR() : theNbins(0) {}

  /// Construct from an already initialized RBorderFinder
  GeneralBinFinderInR(const RBorderFinder& bf) {
    theBorders=bf.RBorders();
    theBins=bf.RBins();
    theNbins=theBins.size();
  }

  /// Construct from the list of Det*
  GeneralBinFinderInR(std::vector<Det*>::const_iterator first,
		      std::vector<Det*>::const_iterator last)
    : theNbins( last-first)
  {
    std::vector<const Det*> dets(first,last);
    RBorderFinder bf(dets);
    theBorders=bf.RBorders();
    theBins=bf.RBins();
    theNbins=theBins.size();
  }

  
  /// Returns an index in the valid range for the bin that contains
  /// AND is closest to R
  int binIndex( T R) const override {
    int i;
    for (i = 0; i<theNbins; i++) {
      if (R < theBorders[i]){
	 break;
      }
    }
    return binIndex(i-1);
  }

  /// Returns an index in the valid range
  int binIndex( int i) const override {
    return std::min( std::max( i, 0), theNbins-1);
  }
   
  /// The middle of the bin
  T binPosition( int ind) const override {
    return theBins[binIndex(ind)];
  }


private:
  int theNbins;
  std::vector<T> theBorders;
  std::vector<T> theBins;

};
#endif


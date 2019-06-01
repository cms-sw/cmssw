#ifndef GeneralBinFinderInZ_H
#define GeneralBinFinderInZ_H

/** \class GeneralBinFinderInZ
 * A Z bin finder for a non-periodic group of detectors.
 */

#include "Utilities/BinningTools/interface/BaseBinFinder.h"
#include "TrackingTools/DetLayers/interface/GeometricSearchDet.h"
#include <cmath>

#include <cassert>

template <class T>
class GeneralBinFinderInZforGeometricSearchDet final : public BaseBinFinder<T> {
public:
  GeneralBinFinderInZforGeometricSearchDet() {}

  GeneralBinFinderInZforGeometricSearchDet(std::vector<const GeometricSearchDet*>::const_iterator first,
                                           std::vector<const GeometricSearchDet*>::const_iterator last)
      : theNbins(last - first - 1)  // -1!
  {
    theBins.reserve(theNbins);
    theBorders.reserve(theNbins);
    for (auto i = first; i < last - 1; i++) {
      theBins.push_back((**i).position().z());
      theBorders.push_back(((**i).position().z() + (**(i + 1)).position().z()) / 2.);
    }
    assert(theNbins == int(theBorders.size()));

    theZOffset = theBorders.front();
    theZStep = (theBorders.back() - theBorders.front()) / (theNbins - 1);
    theInvZStep = 1. / theZStep;
  }

  /// returns an index in the valid range for the bin closest to Z
  int binIndex(T z) const override {
    int bin = int((z - theZOffset) * theInvZStep) + 1;
    if (bin <= 0)
      return 0;
    if (bin >= theNbins)
      return theNbins;

    // check left border
    if (z < theBorders[bin - 1]) {
      // z is to the left of the left border, the correct bin is left
      for (auto i = bin - 1; i > 0; i--) {
        if (z > theBorders[i - 1])
          return i;
      }
      return 0;
    }

    // check right border
    if (z > theBorders[bin]) {
      // z is to the right of the right border, the correct bin is right
      for (int i = bin + 1; i < theNbins; i++) {
        if (z < theBorders[i])
          return i;
      }
      return theNbins;
    }
    // if we arrive here it means that the bin is ok
    return bin;
  }

  /// returns an index in the valid range
  int binIndex(int i) const override { return std::min(std::max(i, 0), theNbins); }

  /// the middle of the bin.
  T binPosition(int ind) const override { return theBins[binIndex(ind)]; }

private:
  int theNbins = 0;  // -1
  T theZStep = 0;
  T theInvZStep = 0;
  T theZOffset = 0;
  std::vector<float> theBorders;
  std::vector<T> theBins;
};
#endif

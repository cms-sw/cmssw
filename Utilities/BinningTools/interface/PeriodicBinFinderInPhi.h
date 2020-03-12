#ifndef PeriodicBinFinderInPhi_H
#define PeriodicBinFinderInPhi_H

#include "Utilities/BinningTools/interface/BaseBinFinder.h"

#include <algorithm>
#include <cmath>

/** Periodic Bin Finder around a circle for (almost) equidistant bins.
 *  Phi is the angle on the circle in radians. 
 */

template <class T>
class PeriodicBinFinderInPhi final : public BaseBinFinder<T> {
public:
  PeriodicBinFinderInPhi() {}

  PeriodicBinFinderInPhi(T firstPhi, int nbins)
      : theNbins(nbins),
        thePhiStep(twoPiC / T(nbins)),
        theInvPhiStep(T(1) / thePhiStep),
        thePhiOffset(firstPhi - T(0.5) * thePhiStep) {}

  /// returns an index in the valid range for the bin that contains phi
  int binIndex(T phi) const override {
    T tmp = std::fmod((phi - thePhiOffset), twoPiC) * theInvPhiStep;
    if (tmp < 0)
      tmp += theNbins;
    return std::min(int(tmp), theNbins - 1);
  }

  /// returns an index in the valid range, modulo Nbins
  int binIndex(int i) const override {
    int ind = i % theNbins;
    return ind < 0 ? ind + theNbins : ind;
  }

  /// the middle of the bin in radians
  T binPosition(int ind) const override { return thePhiOffset + thePhiStep * (T(ind) + T(0.5)); }

  static constexpr T pi() { return piC; }
  static constexpr T twoPi() { return twoPiC; }

private:
  static constexpr T piC = 3.141592653589793238;
  static constexpr T twoPiC = 2 * piC;

  int theNbins = 0;
  T thePhiStep = 0;
  T theInvPhiStep = 0;
  T thePhiOffset = 0;
};
#endif

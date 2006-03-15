#ifndef PeriodicBinFinderInPhi_H
#define PeriodicBinFinderInPhi_H

#include "Utilities/BinningTools/interface/BaseBinFinder.h"

#include <cmath>

/** Periodic Bin Finder around a circle for (almost) equidistant bins.
 *  Phi is the angle on the circle in radians. 
 */

template <class T>
class PeriodicBinFinderInPhi : public BaseBinFinder<T> {
public:

  PeriodicBinFinderInPhi() : theNbins(0), thePhiStep(0), thePhiOffset(0) {}

  PeriodicBinFinderInPhi( T firstPhi, int nbins) :
    theNbins( nbins), thePhiStep( twoPi() / nbins),
    thePhiOffset( firstPhi - thePhiStep/2.) {}

  /// returns an index in the valid range for the bin that contains phi
  virtual  int binIndex( T phi) const {
    T tmp = fmod((phi - thePhiOffset), twoPi()) / thePhiStep;
    if ( tmp < 0) tmp += theNbins;
    return min( int(tmp), theNbins-1);
  }

  /// returns an index in the valid range, modulo Nbins
  virtual int binIndex( int i) const {
    int ind = i % theNbins;
    return ind < 0 ? ind+theNbins : ind;
  }
   
  /// the middle of the bin in radians
  virtual T binPosition( int ind) const {
    return thePhiOffset + thePhiStep * ( ind + 0.5);
  }

  static T pi() { return 3.141592653589793238;}
  static T twoPi() { return 2.*pi();}

private:

  int theNbins;
  T thePhiStep;
  T thePhiOffset;

};
#endif

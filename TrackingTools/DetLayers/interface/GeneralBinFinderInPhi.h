#ifndef TrackingTools_DetLayers_GeneralBinFinderInPhi_H
#define TrackingTools_DetLayers_GeneralBinFinderInPhi_H

/** \class GeneralBinFinderInPhi
 * A phi bin finder for a non-periodic group of detectors.
 *
 *  \author N. Amapane - INFN Torino
 */

#include "Utilities/BinningTools/interface/BaseBinFinder.h"
#include "TrackingTools/DetLayers/interface/PhiBorderFinder.h"
#include "DataFormats/GeometryVector/interface/Pi.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <vector>

template <class T>
class GeneralBinFinderInPhi : public BaseBinFinder<T> {
public:

  typedef PhiBorderFinder::Det Det; //FIXME!!!

  GeneralBinFinderInPhi() : theNbins(0) {}

  /// Construct from an already initialized PhiBorderFinder
  GeneralBinFinderInPhi(const PhiBorderFinder& bf) {
    theBorders=bf.phiBorders();
    theBins=bf.phiBins();
    theNbins=theBins.size();
  }

  /// Construct from the list of Det*
  GeneralBinFinderInPhi(std::vector<Det*>::const_iterator first,
			std::vector<Det*>::const_iterator last)
    : theNbins( last-first)
  {
    std::vector<const Det*> dets(first,last);
    PhiBorderFinder bf(dets);
    theBorders=bf.phiBorders();
    theBins=bf.phiBins();
    theNbins=theBins.size();
  }

  ~GeneralBinFinderInPhi() override{};

  /// Returns an index in the valid range for the bin that contains 
  /// AND is closest to phi
  int binIndex( T phi) const override {
    
    const std::string metname = "Muon|RecoMuon|RecoMuonDetLayers|GeneralBinFinderInPhi";

    static const T epsilon = 10*std::numeric_limits<T>::epsilon();
    // Assume -pi, pi range in pi (which is the case for Geom::Phi

    LogTrace(metname) << "GeneralBinFinderInPhi::binIndex,"
		      << " Nbins: "<< theNbins;

    for (int i = 0; i< theNbins; i++) {

      T cur = theBorders[i];
      T next = theBorders[binIndex(i+1)];
      T phi_ = phi;

      LogTrace(metname) << "bin: " << i 
			<< " border min " << cur << " border max: " << next << " phi: "<< phi_;

      if ( cur > next ) // we are crossing the pi edge: so move the edge to 0!
	{
	  cur = positiveRange(cur);
	  next = positiveRange(next);
	  phi_ = positiveRange(phi_); 
	}
      if (phi_ > cur-epsilon && phi_ < next) return i;
    }
    throw cms::Exception("UnexpectedState") << "GeneralBinFinderInPhi::binIndex( T phi) bin not found!";
  }
  
  /// Returns an index in the valid range, modulo Nbins
  int binIndex( int i) const override {
    int ind = i % (int)theNbins;
    return (ind < 0) ? ind+theNbins : ind;
  }

  /// the middle of the bin in radians
  T binPosition( int ind) const override {
    return theBins[binIndex(ind)];
  }


private:
  int theNbins;
  std::vector<T> theBorders;
  std::vector<T> theBins;

  // returns a positive angle; does NOT reduce the range to 2 pi
  inline T positiveRange (T phi) const
  {
    return (phi > 0) ? phi : phi + Geom::twoPi();
  }

};
#endif


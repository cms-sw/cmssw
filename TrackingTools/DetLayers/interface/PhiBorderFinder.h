#ifndef TrackingTools_DetLayers_PhiBorderFinder_H
#define TrackingTools_DetLayers_PhiBorderFinder_H

/** \class PhiBorderFinder
 *  Find the phi binning of a list of detector according to several 
 *  definitions.
 *
 *  \author N. Amapane - INFN Torino
 */


#include <Geometry/CommonDetUnit/interface/GeomDet.h>

#include <DataFormats/GeometrySurface/interface/BoundingBox.h>
#include <DataFormats/GeometrySurface/interface/GeometricSorting.h>

#include <DataFormats/GeometryVector/interface/Pi.h>
#include <Utilities/General/interface/precomputed_value_sort.h>
#include <TrackingTools/DetLayers/interface/simple_stat.h>
#include <TrackingTools/DetLayers/interface/DetRod.h>
#include <FWCore/Utilities/interface/Exception.h>

// FIXME: remove this include
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <vector>

class PhiBorderFinder {
public:
  
  typedef DetRod Det; //FIXME!!!
  typedef geomsort::ExtractPhi<Det,float> DetPhi;


  PhiBorderFinder(const std::vector<const Det*>& utheDets);
  
  virtual ~PhiBorderFinder(){};

  inline unsigned int nBins() {return theNbins;}

  /// Returns true if the Dets are periodic in phi.
  inline bool isPhiPeriodic() const { return isPhiPeriodic_; }
  
  /// Returns true if any 2 of the Det overlap in phi.
  inline bool isPhiOverlapping() const { return isPhiOverlapping_; }

  /// The borders, defined for each det as the middle between its lower 
  /// edge and the previous Det's upper edge.
  inline const std::vector<double>& phiBorders() const { return thePhiBorders; }

  /// The centers of the Dets.
  inline const std::vector<double>& phiBins() const { return thePhiBins; }

  //  inline std::vector<double> etaBorders() {}
  //  inline std::vector<double> zBorders() {}

private:
  unsigned int theNbins;
  bool isPhiPeriodic_;
  bool isPhiOverlapping_;
  std::vector<double> thePhiBorders;
  std::vector<double> thePhiBins;

  inline double positiveRange (double phi) const
  {
    return (phi > 0) ? phi : phi + Geom::twoPi();
  }

  int binIndex( int i) const {
    int ind = i % (int)theNbins;
    return (ind < 0) ? ind+theNbins : ind;
  }


};
#endif


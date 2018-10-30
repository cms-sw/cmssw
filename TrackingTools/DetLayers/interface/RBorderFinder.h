#ifndef TrackingTools_DetLayers_RBorderFinder_H
#define TrackingTools_DetLayers_RBorderFinder_H

/** \class RBorderFinder
 *  Find the R binning of a list of detector according to several 
 *  definitions.
 *
 *  \author N. Amapane - INFN Torino
 */

#include <DataFormats/GeometrySurface/interface/BoundingBox.h>
#include <DataFormats/GeometrySurface/interface/GeometricSorting.h>

#include <Utilities/General/interface/precomputed_value_sort.h>
#include <Geometry/CommonDetUnit/interface/GeomDet.h>
#include <TrackingTools/DetLayers/interface/ForwardDetRing.h>
#include <TrackingTools/DetLayers/interface/simple_stat.h>
#include <FWCore/Utilities/interface/Exception.h>

#include <vector>

class RBorderFinder {
public:
  
  typedef ForwardDetRing Det; //FIXME!!!
  typedef geomsort::ExtractR<Det,float> DetR;

  RBorderFinder(const std::vector<const Det*>& utheDets);
  
  virtual ~RBorderFinder(){};

  /// Returns true if the Dets are periodic in R.
  inline bool isRPeriodic() const { return isRPeriodic_; }
  
  /// Returns true if any 2 of the Det overlap in R.
  inline bool isROverlapping() const { return isROverlapping_; }

  /// The borders, defined for each det as the middle between its lower 
  /// edge and the previous Det's upper edge.
  inline std::vector<double> RBorders() const { return theRBorders; }

  /// The centers of the Dets.
  inline std::vector<double> RBins() const { return theRBins; }

  //  inline std::vector<double> etaBorders() {}
  //  inline std::vector<double> zBorders() {}


private:
  int theNbins;
  bool isRPeriodic_;
  bool isROverlapping_;
  std::vector<double> theRBorders;
  std::vector<double> theRBins;

  inline int binIndex( int i) const {
    return std::min( std::max( i, 0), theNbins-1);
  }
};
#endif


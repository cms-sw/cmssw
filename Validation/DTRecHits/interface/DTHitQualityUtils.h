#ifndef Validation_DTRecHits_DTHitQualityUtils_h
#define Validation_DTRecHits_DTHitQualityUtils_h

/** \class DTHitQualityUtils
 *  
 *  Define some basic tools and utilities for 1D DT Rec Hit and 
 *  2D, 4D DT Segment analysis
 *
 *  \author S. Bolognesi and G. Cerminara - INFN Torino
 */

#include "DataFormats/GeometryVector/interface/LocalVector.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

#include <map>
#include <atomic>

class PSimHit;
class DTGeometry;

namespace DTHitQualityUtils {

  /// Operations
  /// Create a map between the SimHits in a chamber and the corrisponding MuBarWireId
  std::map<DTWireId, edm::PSimHitContainer > mapSimHitsPerWire(const edm::PSimHitContainer& simhits) ;
  /// Create a map between the Mu SimHits and corresponding MuBarWireId ;
  std::map<DTWireId, const PSimHit*> mapMuSimHitsPerWire(const std::map<DTWireId, edm::PSimHitContainer>& simHitWireMap) ;
  /// Select the SimHit from a muon in a vector of SimHits
  const PSimHit* findMuSimHit(const edm::PSimHitContainer& hits); 
  /// Find Innermost and outermost SimHit from Mu in a SL (they identify a simulated segment)
  std::pair<const PSimHit*, const PSimHit*> findMuSimSegment(const std::map<DTWireId, const PSimHit*>& mapWireAndMuSimHit) ;
  /// Find direction and position of a segment (in local RF) from outer and inner mu SimHit in the RF of object Det
  std::pair<LocalVector, LocalPoint> findMuSimSegmentDirAndPos(const std::pair<const PSimHit*, const PSimHit*>& inAndOutSimHit, 
								      const DetId detId, const DTGeometry *muonGeom);
  /// Find the angles from a segment direction:
  /// atan(dx/dz) = "phi"   angle in the chamber RF
  /// atan(dy/dz) = "theta" angle in the chamber RF (note: this has opposite sign in the SLZ RF!)
  std::pair<double, double> findSegmentAlphaAndBeta(const LocalVector& direction);

  // Set the verbosity level
  extern std::atomic<bool> debug; 

  //Find angle error
  double sigmaAngle(double Angle, double sigma2TanAngle);

}
#endif

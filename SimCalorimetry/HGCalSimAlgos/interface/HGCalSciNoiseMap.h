#ifndef simcalorimetry_hgcalsimalgos_hgcalscinoisemap
#define simcalorimetry_hgcalsimalgos_hgcalscinoisemap

#include "SimCalorimetry/HGCalSimAlgos/interface/HGCalRadiationMap.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include <string>


/**
   @class HGCalSciNoiseMap
   @short derives from HGCalRadiation map to parse fluence parameters, provides Sci-specific functions
*/
class HGCalSciNoiseMap : public HGCalRadiationMap {

 public:

  HGCalSciNoiseMap();
  ~HGCalSciNoiseMap() {};

  /**
     @short returns the signal scaling and the noise
  */
  double scaleByArea(const HGCScintillatorDetId&, const std::array<double, 8>&);
  std::pair<double, double> scaleByDose(const HGCScintillatorDetId&, const std::array<double, 8>&);

  std::array<double, 8> computeRadius(const HGCScintillatorDetId&);


 private:

  //size of the reference scintillator tile
  const double refEdge_;

};

#endif

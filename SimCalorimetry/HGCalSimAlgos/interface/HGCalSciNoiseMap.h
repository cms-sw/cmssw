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
  ~HGCalSciNoiseMap(){};

  /**
     @short returns the signal scaling and the noise
  */
  double scaleByTileArea(const HGCScintillatorDetId&, const radiiVec&);
  double scaleBySipmArea(const HGCScintillatorDetId&, const double&);
  std::pair<double, double> scaleByDose(const HGCScintillatorDetId&, const radiiVec&);

  radiiVec computeRadius(const HGCScintillatorDetId&);
  void setSipmMap(const std::string&);

private:
  std::unordered_map<int, float> readSipmPars(const std::string&);

  //size of the reference scintillator tile
  const double refEdge_;
  //sipm size boundaries
  std::unordered_map<int, float> sipmMap_;
};

#endif

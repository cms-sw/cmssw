#ifndef simcalorimetry_hgcalsimalgos_hgcalscinoisemap
#define simcalorimetry_hgcalsimalgos_hgcalscinoisemap

#include "SimCalorimetry/HGCalSimAlgos/interface/HGCalRadiationMap.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include <string>

/**
   @class HGCalSciNoiseMap
   @short derives from HGCalRadiation map to parse fluence/dose parameters, provides Sci-specific functions
          the algo word (set at configuration level) is used to control which aspects are simulated
          bit 1 - ignores the scaling of signal and noise with SIPMAREA
          bit 2 - instead of the geometry-based SiPM area (from detId, if available) use the boundaries read from a txt file
          bit 3 - ignores the scaling of the signal light yield with the tile area
          bit 4 - ignores the scaling of the light yield with the dose
          bit 5 - ignores the scaling of the noise with the fluence (=constant noise scenario)
          bit 6 - ignores noise
*/
class HGCalSciNoiseMap : public HGCalRadiationMap {
public:

  enum NoiseMapAlgoBits_t { IGNORE_SIPMAREA, OVERRIDE_SIPMAREA, IGNORE_TILEAREA, IGNORE_DOSESCALE, IGNORE_FLUENCESCALE, IGNORE_NOISE };
  HGCalSciNoiseMap();
  ~HGCalSciNoiseMap(){};

  /**
     @short returns the signal scaling and the noise
  */
  double scaleByTileArea(const HGCScintillatorDetId &, const double);
  double scaleBySipmArea(const HGCScintillatorDetId &, const double);
  std::pair<double, double> scaleByDose(const HGCScintillatorDetId &, const double);

  void setDoseMap(const std::string &,const unsigned int );
  void setSipmMap(const std::string &);
  void setReferenceDarkCurrent(double idark);

private:

  /**
     @short parses the radius boundaries for the SiPM area assignment from a custom file
   */
  std::unordered_map<int, float> readSipmPars(const std::string &);

  //size of the reference scintillator tile
  const double refEdge_;

  //flags used to disable/override specific SiPM-on-tile operation parameters
  bool ignoreSiPMarea_, overrideSiPMarea_, ignoreTileArea_, ignoreDoseScale_, ignoreFluenceScale_, ignoreNoise_;

  //reference dark current for the noise (mA)
  double refDarkCurrent_;

  //sipm size boundaries
  std::unordered_map<int, float> sipmMap_;
};

#endif

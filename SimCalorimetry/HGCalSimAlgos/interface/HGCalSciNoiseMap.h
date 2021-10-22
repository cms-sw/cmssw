#ifndef simcalorimetry_hgcalsimalgos_hgcalscinoisemap
#define simcalorimetry_hgcalsimalgos_hgcalscinoisemap

#include "SimCalorimetry/HGCalSimAlgos/interface/HGCalRadiationMap.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include <string>
#include <array>

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
          bit 7 - ignore tile type (fallback on CAST)
          bit 8 - ignore pedestal subtraction
*/

class HGCalSciNoiseMap : public HGCalRadiationMap {
public:
  enum TileType_t { CAST, MOULDED, TILETYPE_N };
  enum GainRange_t { GAIN_2, GAIN_4, AUTO, GAINRANGE_N };  //roc gain for 2mm2 and 4mm2
  enum NoiseMapAlgoBits_t {
    IGNORE_SIPMAREA,
    OVERRIDE_SIPMAREA,
    IGNORE_TILEAREA,
    IGNORE_DOSESCALE,
    IGNORE_FLUENCESCALE,
    IGNORE_NOISE,
    IGNORE_TILETYPE,
    IGNORE_AUTO_PEDESTALSUB
  };

  struct SiPMonTileCharacteristics {
    SiPMonTileCharacteristics() : s(0.), lySF(0.), n(0.), xtalk(0), gain(0), thrADC(0), ntotalPE(0) {}
    float s, lySF, n, xtalk;
    unsigned short gain, thrADC, ntotalPE;
  };

  HGCalSciNoiseMap();
  ~HGCalSciNoiseMap(){};

  /**
     @short returns the signal scaling and the noise
  */
  double scaleByTileArea(const HGCScintillatorDetId &, const double);
  std::pair<double, GainRange_t> scaleBySipmArea(const HGCScintillatorDetId &, const double, const GainRange_t &);
  SiPMonTileCharacteristics scaleByDose(const HGCScintillatorDetId &,
                                        const double,
                                        const int aimMIPtoADC = 15,
                                        const GainRange_t gainPreChoice = GainRange_t::AUTO);

  void setDoseMap(const std::string &, const unsigned int);
  void setSipmMap(const std::string &);
  void setReferenceDarkCurrent(double idark);
  void setReferenceCrossTalk(double xtalk) { refXtalk_ = xtalk; }
  void setNpePerMIP(float npePerMIP);
  std::array<double, GAINRANGE_N> getLSBPerGain() { return lsbPerGain_; }
  std::array<double, GAINRANGE_N> getMaxADCPerGain() { return fscADCPerGain_; }
  std::array<double, TILETYPE_N> getNpePerMIP() { return nPEperMIP_; }
  float getNPeInSiPM() { return maxSiPMPE_; }
  bool ignoreAutoPedestalSubtraction() { return ignoreAutoPedestalSub_; }

private:
  /**
     @short parses the radius boundaries for the SiPM area assignment from a custom file
   */
  std::unordered_map<int, float> readSipmPars(const std::string &);

  //reference signal yields
  std::array<double, TILETYPE_N> nPEperMIP_;

  //lsb and fsc per gain
  std::array<double, GAINRANGE_N> lsbPerGain_, fscADCPerGain_;

  //size of the reference scintillator tile
  const double refEdge_;

  //flags used to disable/override specific SiPM-on-tile operation parameters
  bool ignoreSiPMarea_, overrideSiPMarea_, ignoreTileArea_, ignoreDoseScale_, ignoreFluenceScale_, ignoreNoise_,
      ignoreTileType_, ignoreAutoPedestalSub_;

  //reference dark current for the noise (mA)
  double refDarkCurrent_;

  //reference cross talk parameter (0,1) - if -1 it will be used to ignore effect in the digitization step
  double refXtalk_;

  //reference ADC counts for the MIP peak
  int aimMIPtoADC_;

  //reference number of pixels (2mm2 SiPM)
  float maxSiPMPE_;

  //sipm size boundaries
  std::unordered_map<int, float> sipmMap_;
};

#endif

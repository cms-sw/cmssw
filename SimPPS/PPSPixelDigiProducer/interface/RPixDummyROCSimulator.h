#ifndef RPIX_DUMMY_ROC_SIMULATION_H
#define RPIX_DUMMY_ROC_SIMULATION_H

#include <set>

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/CTPPSDigi/interface/CTPPSPixelDigi.h"
#include "DataFormats/CTPPSDigi/interface/CTPPSPixelDigiCollection.h"
#include "CondFormats/PPSObjects/interface/CTPPSPixelGainCalibrations.h"
#include "CondFormats/PPSObjects/interface/CTPPSPixelAnalysisMask.h"

class RPixDummyROCSimulator {
public:
  RPixDummyROCSimulator(const edm::ParameterSet &params, uint32_t det_id);

  void ConvertChargeToHits(const std::map<unsigned short, double> &signals,
                           std::map<unsigned short, std::vector<std::pair<int, double> > > &theSignalProvenance,
                           std::vector<CTPPSPixelDigi> &output_digi,
                           std::vector<std::vector<std::pair<int, double> > > &output_digi_links,
                           const CTPPSPixelGainCalibrations *pcalibration);

private:
  typedef std::set<unsigned short> dead_pixel_set;
  static constexpr double highRangeCal_ = 1800.;
  static constexpr double lowRangeCal_ = 260.;
  static constexpr int maxADC_ = 255;

  uint32_t det_id_;
  double dead_pixel_probability_;
  bool dead_pixels_simulation_on_;
  dead_pixel_set dead_pixels_;
  int verbosity_;
  unsigned short pixels_no_;
  double threshold_;
  double electron_per_adc_;
  int VcaltoElectronGain_;
  int VcaltoElectronOffset_;
  bool doSingleCalibration_;
  bool links_persistence_;
};

#endif

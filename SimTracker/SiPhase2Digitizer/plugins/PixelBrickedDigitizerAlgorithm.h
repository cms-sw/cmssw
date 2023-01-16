
#ifndef _SimTracker_SiPhase2Digitizer_PixelBrickedDigitizerAlgorithm_h
#define _SimTracker_SiPhase2Digitizer_PixelBrickedDigitizerAlgorithm_h

#include "CondFormats/SiPixelObjects/interface/GlobalPixel.h"
#include "CondFormats/DataRecord/interface/SiPixelQualityRcd.h"
#include "CondFormats/DataRecord/interface/SiPixelFedCablingMapRcd.h"
#include "CondFormats/DataRecord/interface/SiPixelLorentzAngleSimRcd.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "SimTracker/SiPhase2Digitizer/plugins/PixelDigitizerAlgorithm.h"

class PixelBrickedDigitizerAlgorithm : public PixelDigitizerAlgorithm {
private:
public:
  PixelBrickedDigitizerAlgorithm(const edm::ParameterSet& conf, edm::ConsumesCollector iC);
  ~PixelBrickedDigitizerAlgorithm() override;

  // Specific for bricked pixel
  void induce_signal(std::vector<PSimHit>::const_iterator inputBegin,
                     const PSimHit& hit,
                     const size_t hitIndex,
                     const size_t firstHitIndex,
                     const unsigned int tofBin,
                     const Phase2TrackerGeomDetUnit* pixdet,
                     const std::vector<digitizerUtility::SignalPoint>& collection_points) override;
};
#endif

#ifndef TrackingAnalysis_PixelPSimHitSelector_h
#define TrackingAnalysis_PixelPSimHitSelector_h

#include "CondFormats/DataRecord/interface/SiPixelQualityRcd.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelQuality.h"

#include "SimGeneral/TrackingAnalysis/interface/PSimHitSelector.h"

//! PixelPSimHitSelector class
class PixelPSimHitSelector : public PSimHitSelector {
public:
  //! Constructor by pset.
  /* Creates a PixelPSimHitSelector with association given by pset.

     /param[in] pset with the configuration values
  */
  PixelPSimHitSelector(edm::ParameterSet const &config, edm::ConsumesCollector &iC)
      : PSimHitSelector(config, iC), badModuleToken_(iC.esConsumes()) {}

  //! Pre-process event information
  void select(PSimHitCollection &, edm::Event const &, edm::EventSetup const &) const override;

private:
  edm::ESGetToken<SiPixelQuality, SiPixelQualityRcd> badModuleToken_;
};

#endif

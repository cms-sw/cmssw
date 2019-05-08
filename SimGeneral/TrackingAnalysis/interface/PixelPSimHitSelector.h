#ifndef TrackingAnalysis_PixelPSimHitSelector_h
#define TrackingAnalysis_PixelPSimHitSelector_h

#include "SimGeneral/TrackingAnalysis/interface/PSimHitSelector.h"

//! PixelPSimHitSelector class
class PixelPSimHitSelector : public PSimHitSelector {
public:
  //! Constructor by pset.
  /* Creates a PixelPSimHitSelector with association given by pset.

     /param[in] pset with the configuration values
  */
  PixelPSimHitSelector(edm::ParameterSet const &config) : PSimHitSelector(config) {}

  //! Pre-process event information
  void select(PSimHitCollection &, edm::Event const &, edm::EventSetup const &) const override;
};

#endif

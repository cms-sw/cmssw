// This is a plugin implementation, but it is in src/ to make it possible to 
// derive from it in other packages. In plugins/ there is a dummy that declares
// the plugin.
#include "Validation/SiPixelPhase1CommonV/interface/SiPixelPhase1BaseV.h"
#include "FWCore/Framework/interface/MakerMacros.h"

void SiPixelPhase1HarvesterV::dqmEndLuminosityBlock(DQMStore::IBooker& iBooker, DQMStore::IGetter& iGetter, edm::LuminosityBlock const& lumiBlock, edm::EventSetup const& eSetup) {
  for (HistogramManager& histoman : histo)
    histoman.executeHarvestingOnline(iBooker, iGetter, eSetup);
};
void SiPixelPhase1HarvesterV::dqmEndJob(DQMStore::IBooker& iBooker, DQMStore::IGetter& iGetter) {
  for (HistogramManager& histoman : histo)
    histoman.executeHarvestingOffline(iBooker, iGetter);
};

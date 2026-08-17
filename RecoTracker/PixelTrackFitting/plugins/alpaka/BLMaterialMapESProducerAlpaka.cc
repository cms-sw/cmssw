#include <memory>

#include <alpaka/alpaka.hpp>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ModuleFactory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/PixelTrackFitting/interface/BLMaterialMapHost.h"
#include "RecoTracker/PixelTrackFitting/interface/alpaka/BLMaterialMapCollection.h"
#include "RecoTracker/Record/interface/BLMaterialMapRecord.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // Provides the BL-fit Geant4 material map rho(r,z) as an EventSetup portable condition: produces the host
  // payload once (BLMaterialMapHost, filled from the compiled-in D121 table blMaterialMapDataD121()); copied to
  // the device once per IOV (CopyToDevice<BLMaterialMapHost> in BLMaterialMapCollection.h). The table is the
  // material model the GBL fits are calibrated against; a different material model needs a different table.
  class BLMaterialMapESProducerAlpaka : public ESProducer {
  public:
    BLMaterialMapESProducerAlpaka(edm::ParameterSet const& iConfig) : ESProducer(iConfig) { setWhatProduced(this); }

    std::unique_ptr<BLMaterialMapHost> produce(const BLMaterialMapRecord& /*iRecord*/) {
      // BLMaterialMapHost's constructor fills the host buffer from the compiled-in table.
      return std::make_unique<BLMaterialMapHost>();
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      descriptions.addWithDefaultLabel(desc);
    }
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(BLMaterialMapESProducerAlpaka);

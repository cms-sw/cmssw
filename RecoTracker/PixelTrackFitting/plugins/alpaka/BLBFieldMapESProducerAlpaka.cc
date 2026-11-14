#include <memory>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ModuleFactory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "RecoTracker/PixelTrackFitting/interface/BLBFieldMap.h"
#include "RecoTracker/PixelTrackFitting/interface/BLBFieldMapHost.h"
#include "RecoTracker/PixelTrackFitting/interface/alpaka/BLBFieldMapDevice.h"
#include "RecoTracker/Record/interface/BLBFieldMapRecord.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // The (Bz, Br) field map of the BrokenLine fit. Sample the MagneticField product at phi = 0 on the blBFieldMap
  // lattice, once per IOV, normalized to Bz(0,0). The field is taken as axisymmetric.
  class BLBFieldMapESProducerAlpaka : public ESProducer {
  public:
    BLBFieldMapESProducerAlpaka(edm::ParameterSet const& iConfig) : ESProducer(iConfig) {
      auto cc = setWhatProduced(this);
      fieldToken_ = cc.consumes();
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      descriptions.addWithDefaultLabel(desc);
    }

    std::unique_ptr<BLBFieldMapHost> produce(const BLBFieldMapRecord& iRecord) {
      MagneticField const& field = iRecord.get(fieldToken_);

      const double bz00 = field.inTesla(GlobalPoint(0.f, 0.f, 0.f)).z();
      // With no field at the origin, store the samples in tesla.
      const double norm = (bz00 != 0.0) ? bz00 : 1.0;

      auto product = std::make_unique<BLBFieldMapHost>(cms::alpakatools::host());
      float* const bz = product->value().bz;
      float* const br = product->value().br;
      for (int ir = 0; ir < blBFieldMap::kNR; ++ir) {
        const double r = blBFieldMap::nodeR(ir);
        for (int iz = 0; iz < blBFieldMap::kNZ; ++iz) {
          const double z = blBFieldMap::nodeZ(iz);
          const auto b = field.inTesla(GlobalPoint(float(r), 0.f, float(z)));
          bz[blBFieldMap::nodeIndex(ir, iz)] = float(b.z() / norm);
          br[blBFieldMap::nodeIndex(ir, iz)] = float(b.x() / norm);
        }
      }

      // Log the origin field and two ratios, to show a wrong map.
      const int izMax = blBFieldMap::kNZ - 1;
      edm::LogInfo("BLBFieldMap") << "(Bz,Br)/Bz(0,0) map on " << blBFieldMap::kNR << "x" << blBFieldMap::kNZ
                                  << " (r,z) nodes: Bz(0,0)=" << bz00
                                  << " T, Bz(0,+zMax)/Bz(0,0)=" << bz[blBFieldMap::nodeIndex(0, izMax)]
                                  << ", Bz(rMax,0)/Bz(0,0)="
                                  << bz[blBFieldMap::nodeIndex(blBFieldMap::kNR - 1, izMax / 2)];

      return product;
    }

  private:
    edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> fieldToken_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(BLBFieldMapESProducerAlpaka);

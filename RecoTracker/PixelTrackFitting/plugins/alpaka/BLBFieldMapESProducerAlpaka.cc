#include <algorithm>
#include <array>
#include <cmath>
#include <memory>

#include <alpaka/alpaka.hpp>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ModuleFactory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "RecoTracker/PixelTrackFitting/interface/BLBFieldMap.h"
#include "RecoTracker/PixelTrackFitting/interface/BLBFieldMapHost.h"
#include "RecoTracker/PixelTrackFitting/interface/alpaka/BLBFieldMapCollection.h"
#include "RecoTracker/Record/interface/BLBFieldMapRecord.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // Provides the BL-fit (Bz,Br) field map as an EventSetup portable condition: samples the MagneticField
  // product on the blBFieldMap lattice once per IOV, both components normalized to Bz(0,0). Sampling at
  // phi = 0 suffices, the solenoid map being axisymmetric to the precision the conversion needs.
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

      const double dR = double(blBFieldMap::kRMax) / double(blBFieldMap::kNR - 1);
      const double dZ = 2.0 * double(blBFieldMap::kZMax) / double(blBFieldMap::kNZ - 1);
      const double bz00 = field.inTesla(GlobalPoint(0.f, 0.f, 0.f)).z();
      // No field: flat unit Bz and zero Br, i.e. the effective field everywhere equals the field at the origin.
      const double norm = (bz00 != 0.0) ? bz00 : 1.0;

      std::array<float, blBFieldMap::kNValues> values{};
      for (int ir = 0; ir < blBFieldMap::kNR; ++ir) {
        const double r = dR * double(ir);
        for (int iz = 0; iz < blBFieldMap::kNZ; ++iz) {
          const double z = -double(blBFieldMap::kZMax) + dZ * double(iz);
          const auto b = field.inTesla(GlobalPoint(float(r), 0.f, float(z)));
          values[ir * blBFieldMap::kNZ + iz] = float(b.z() / norm);
          values[blBFieldMap::kNNodes + ir * blBFieldMap::kNZ + iz] = float(b.x() / norm);
        }
      }

      // Origin field, on-axis |z| falloff and largest radial ratio, so a wrong or absent map is visible in the log.
      const int izMax = blBFieldMap::kNZ - 1;
      double maxBrRatio = 0.0;
      for (int i = blBFieldMap::kNNodes; i < blBFieldMap::kNValues; ++i)
        maxBrRatio = std::max(maxBrRatio, std::abs(double(values[i])));
      edm::LogInfo("BLBFieldMap") << "(Bz,Br)/Bz(0,0) map on " << blBFieldMap::kNR << "x" << blBFieldMap::kNZ
                                  << " (r,z) nodes: Bz(0,0)=" << bz00 << " T, Bz(0,+zMax)/Bz(0,0)=" << values[izMax]
                                  << ", Bz(rMax,0)/Bz(0,0)="
                                  << values[(blBFieldMap::kNR - 1) * blBFieldMap::kNZ + izMax / 2]
                                  << ", max |Br|/Bz(0,0)=" << maxBrRatio;

      return std::make_unique<BLBFieldMapHost>(values);
    }

  private:
    edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> fieldToken_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(BLBFieldMapESProducerAlpaka);

// Checks the two EventSetup maps read by the BrokenLine fits, and optionally writes them out:
// the field map must match the MagneticField sampled on the same lattice and normalization as
// BLBFieldMapESProducerAlpaka, the material map must match the compiled-in D121 table. Bit for bit; a
// mismatch throws.
#include <array>
#include <cmath>
#include <fstream>
#include <string>

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoTracker/PixelTrackFitting/interface/BLBFieldMap.h"
#include "RecoTracker/PixelTrackFitting/interface/BLBFieldMapHost.h"
#include "RecoTracker/PixelTrackFitting/interface/BLMaterialMap.h"
#include "RecoTracker/PixelTrackFitting/interface/BLMaterialMapHost.h"
#include "RecoTracker/Record/interface/BLBFieldMapRecord.h"
#include "RecoTracker/Record/interface/BLMaterialMapRecord.h"

class BLMapsCheck : public edm::one::EDAnalyzer<> {
public:
  explicit BLMapsCheck(edm::ParameterSet const& iConfig)
      : fieldMapToken_(esConsumes()),
        materialMapToken_(esConsumes()),
        fieldToken_(esConsumes()),
        outputFile_(iConfig.getUntrackedParameter<std::string>("outputFile")) {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.addUntracked<std::string>("outputFile", "")
        ->setComment("if set, both lattices are written to this text file, one value per line");
    descriptions.addWithDefaultLabel(desc);
  }

  void analyze(edm::Event const&, edm::EventSetup const& iSetup) override {
    // field map: recompute from the MagneticField exactly as the ESProducer does
    MagneticField const& field = iSetup.getData(fieldToken_);
    const double dR = double(blBFieldMap::kRMax) / double(blBFieldMap::kNR - 1);
    const double dZ = 2.0 * double(blBFieldMap::kZMax) / double(blBFieldMap::kNZ - 1);
    const double bz00 = field.inTesla(GlobalPoint(0.f, 0.f, 0.f)).z();
    const double norm = (bz00 != 0.0) ? bz00 : 1.0;
    std::array<float, blBFieldMap::kNValues> expected{};
    for (int ir = 0; ir < blBFieldMap::kNR; ++ir) {
      const double r = dR * double(ir);
      for (int iz = 0; iz < blBFieldMap::kNZ; ++iz) {
        const double z = -double(blBFieldMap::kZMax) + dZ * double(iz);
        const auto b = field.inTesla(GlobalPoint(float(r), 0.f, float(z)));
        expected[ir * blBFieldMap::kNZ + iz] = float(b.z() / norm);
        expected[blBFieldMap::kNNodes + ir * blBFieldMap::kNZ + iz] = float(b.x() / norm);
      }
    }
    float const* fieldMap = iSetup.getData(fieldMapToken_).data();
    int nBad = 0;
    for (int i = 0; i < blBFieldMap::kNValues; ++i)
      nBad += (fieldMap[i] != expected[i]);
    if (nBad)
      throw cms::Exception("BLMapsCheck") << nBad << " of " << blBFieldMap::kNValues
                                          << " field-map values differ from the MagneticField sampled on the lattice";

    // material map: must be the compiled-in table
    float const* materialMap = iSetup.getData(materialMapToken_).data();
    float const* table = blMaterialMap::blMaterialMapData();
    nBad = 0;
    for (int i = 0; i < blMaterialMap::kSize; ++i)
      nBad += (materialMap[i] != table[i]);
    if (nBad)
      throw cms::Exception("BLMapsCheck")
          << nBad << " of " << blMaterialMap::kSize << " material-map values differ from the compiled-in table";

    edm::LogPrint("BLMapsCheck") << "field map: " << blBFieldMap::kNValues
                                 << " values reproduced from the MagneticField (Bz(0,0) = " << bz00
                                 << " T); material map: " << blMaterialMap::kSize
                                 << " values identical to the compiled-in table";

    if (!outputFile_.empty()) {
      std::ofstream out(outputFile_);
      out.precision(9);
      out << "# BLBFieldMap: " << blBFieldMap::kNR << " x " << blBFieldMap::kNZ
          << " nodes, Bz block then Br block, normalized to Bz(0,0) = " << bz00 << " T\n";
      for (int i = 0; i < blBFieldMap::kNValues; ++i)
        out << fieldMap[i] << "\n";
      out << "# BLMaterialMap: " << blMaterialMap::kNR << " x " << blMaterialMap::kNZ
          << " cells, rho(r,z) [X0/cm], index ir*kNZ+iz\n";
      for (int i = 0; i < blMaterialMap::kSize; ++i)
        out << materialMap[i] << "\n";
    }
  }

private:
  const edm::ESGetToken<BLBFieldMapHost, BLBFieldMapRecord> fieldMapToken_;
  const edm::ESGetToken<BLMaterialMapHost, BLMaterialMapRecord> materialMapToken_;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> fieldToken_;
  const std::string outputFile_;
};

DEFINE_FWK_MODULE(BLMapsCheck);

// TripletFeaturesTableProducer - nano flat table with the per-BUILT-TRIPLET feature rows captured
// on device by the CA_TRIPLET_DUMP path (Kernel_connect) and surfaced as the 'Triplet' product by
// CAHitNtuplet. The robust ntuple route for the in-kernel triplet-DNN training dataset: one row per
// built triplet, joined offline to the per-merged-hit truth (HitTruthTableProducer) by the three
// merged-hit indices h1/h2/h3.
//
// Row space: rows = view().nValid() (the valid-row-count scalar stamped device-side == the number
// of triplets Kernel_connect actually wrote). Only the first nValid rows carry valid data, so we
// emit exactly nValid rows.
//
// Columns mirror caStructures::TripletDumpSoA EXACTLY (18 BASE DNN features in train_triplet_dnn.py
// order, the three CA layer ids, and the three merged-hit join keys).

#include <vector>

#include "DataFormats/NanoAOD/interface/FlatTable.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "RecoTracker/PixelSeeding/interface/TripletDumpHost.h"

class TripletFeaturesTableProducer : public edm::global::EDProducer<> {
public:
  explicit TripletFeaturesTableProducer(const edm::ParameterSet& params)
      : tableName_(params.getParameter<std::string>("tableName")),
        dumpToken_(consumes<caStructures::TripletDumpHost>(params.getParameter<edm::InputTag>("tripletDumpSrc"))) {
    produces<nanoaod::FlatTable>(tableName_);
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<std::string>("tableName", "Triplet");
    // The CA producer emits the dump with produces() (no instance label); the device->host
    // transcription keeps the same module label + empty instance, so just the module name here.
    desc.add<edm::InputTag>("tripletDumpSrc", edm::InputTag("hltPhase2PixelTracksSoALowPt"));
    descriptions.addWithDefaultLabel(desc);
  }

private:
  void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup&) const override {
    const auto& dumpHost = iEvent.get(dumpToken_);
    const auto view = dumpHost.const_view();

    // Valid rows = the count Kernel_connect wrote (full-capacity buffer; rows >= nValid are garbage).
    // Clamp to the allocated capacity for safety.
    const uint32_t cap = static_cast<uint32_t>(view.metadata().size());
    const uint32_t nValid = std::min<uint32_t>(view.nValid(), cap);

    // 18 BASE features (== train_triplet_dnn.py BASE_FEATURES order).
    std::vector<float> absCurvature(nValid), tipTimesCurvature(nValid), dca(nValid), curvatureStubs(nValid),
        curvatureStubsErrSquared(nValid), curvature13(nValid), dPhi12(nValid), dPhi13(nValid), dPhi23(nValid),
        dr12(nValid), dr13(nValid), r1(nValid), r2(nValid), r3(nValid), z1(nValid), z2(nValid), z3(nValid),
        nStubs(nValid), curvature(nValid), inKernelScore(nValid);
    // CA layer ids (int) and the three merged-hit join keys (uint32).
    std::vector<int> lay1(nValid), lay2(nValid), lay3(nValid);
    std::vector<uint32_t> h1(nValid), h2(nValid), h3(nValid);

    for (uint32_t i = 0; i < nValid; ++i) {
      const auto row = view[i];
      absCurvature[i] = row.absCurvature();
      tipTimesCurvature[i] = row.tipTimesCurvature();
      dca[i] = row.dca();
      curvatureStubs[i] = row.curvatureStubs();
      curvatureStubsErrSquared[i] = row.curvatureStubsErrSquared();
      curvature13[i] = row.curvature13();
      dPhi12[i] = row.dPhi12();
      dPhi13[i] = row.dPhi13();
      dPhi23[i] = row.dPhi23();
      dr12[i] = row.dr12();
      dr13[i] = row.dr13();
      r1[i] = row.r1();
      r2[i] = row.r2();
      r3[i] = row.r3();
      z1[i] = row.z1();
      z2[i] = row.z2();
      z3[i] = row.z3();
      nStubs[i] = row.nStubs();
      curvature[i] = row.curvature();
      inKernelScore[i] = row.inKernelScore();
      lay1[i] = row.lay1();
      lay2[i] = row.lay2();
      lay3[i] = row.lay3();
      h1[i] = row.h1();
      h2[i] = row.h2();
      h3[i] = row.h3();
    }

    auto table = std::make_unique<nanoaod::FlatTable>(nValid, tableName_, /*singleton*/ false, /*extension*/ false);
    table->addColumn<float>("absCurvature", absCurvature, "triplet DNN base feature", -1);
    table->addColumn<float>("tipTimesCurvature", tipTimesCurvature, "triplet DNN base feature", -1);
    table->addColumn<float>("dca", dca, "triplet DNN base feature", -1);
    table->addColumn<float>("curvatureStubs", curvatureStubs, "triplet DNN base feature", -1);
    table->addColumn<float>("curvatureStubsErrSquared", curvatureStubsErrSquared, "triplet DNN base feature", -1);
    table->addColumn<float>("curvature13", curvature13, "triplet DNN base feature", -1);
    table->addColumn<float>("curvature", curvature, "signed triplet curvature (for derived features)", -1);
    table->addColumn<float>("inKernelScore", inKernelScore, "in-kernel DNN score<BANK>(feat), -1 if not eval", -1);
    table->addColumn<float>("dPhi12", dPhi12, "triplet DNN base feature", -1);
    table->addColumn<float>("dPhi13", dPhi13, "triplet DNN base feature", -1);
    table->addColumn<float>("dPhi23", dPhi23, "triplet DNN base feature", -1);
    table->addColumn<float>("dr12", dr12, "triplet DNN base feature", -1);
    table->addColumn<float>("dr13", dr13, "triplet DNN base feature", -1);
    table->addColumn<float>("r1", r1, "inner-hit r [cm]", -1);
    table->addColumn<float>("r2", r2, "middle-hit r [cm]", -1);
    table->addColumn<float>("r3", r3, "outer-hit r [cm]", -1);
    table->addColumn<float>("z1", z1, "inner-hit z [cm]", -1);
    table->addColumn<float>("z2", z2, "middle-hit z [cm]", -1);
    table->addColumn<float>("z3", z3, "outer-hit z [cm]", -1);
    table->addColumn<float>("nStubs", nStubs, "number of stub hits in the triplet", -1);
    table->addColumn<int>("lay1", lay1, "inner-hit CA layer id", -1);
    table->addColumn<int>("lay2", lay2, "middle-hit CA layer id", -1);
    table->addColumn<int>("lay3", lay3, "outer-hit CA layer id", -1);
    table->addColumn<uint32_t>("h1", h1, "inner merged-hit index (truth join key)", -1);
    table->addColumn<uint32_t>("h2", h2, "middle merged-hit index (truth join key)", -1);
    table->addColumn<uint32_t>("h3", h3, "outer merged-hit index (truth join key)", -1);
    iEvent.put(std::move(table), tableName_);
  }

  const std::string tableName_;
  const edm::EDGetTokenT<caStructures::TripletDumpHost> dumpToken_;
};

DEFINE_FWK_MODULE(TripletFeaturesTableProducer);

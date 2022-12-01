#include "catch.hpp"
#include "SimG4CMS/Calo/interface/HFShowerLibrary.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "FWCore/PluginManager/interface/PresenceFactory.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "CLHEP/Random/MixMaxRng.h"
#include "Randomize.hh"

namespace {
  std::shared_ptr<edm::Presence> setupMessageLogger(bool iDoSetup) {
    if (not iDoSetup) {
      return std::shared_ptr<edm::Presence>();
    }
    // Initialise the plug-in manager.
    edmplugin::PluginManager::configure(edmplugin::standard::config());

    try {
      return std::shared_ptr<edm::Presence>(
          edm::PresenceFactory::get()->makePresence("SingleThreadMSPresence").release());
    } catch (cms::Exception &e) {
      std::cerr << e.explainSelf() << std::endl;
      throw;
    }
  }

  class SetRandomEngine {
  public:
    explicit SetRandomEngine(long iSeed) : m_current(iSeed), m_previous(G4Random::getTheEngine()) {
      G4Random::setTheEngine(&m_current);
    }
    ~SetRandomEngine() { G4Random::setTheEngine(m_previous); }

  private:
    CLHEP::MixMaxRng m_current;
    decltype(G4Random::getTheEngine()) m_previous;
  };
}  // namespace
namespace test_hffibre {
  HFFibre::Params defaultParams();
}

TEST_CASE("test HFShowerLibrary", "[HFShowerLibrary]") {
  //pass 'true' to turn on MessageLogger output
  static std::shared_ptr<edm::Presence> gobbleUpTheGoop = setupMessageLogger(false);

  auto fibreParams = test_hffibre::defaultParams();

  //These values come from IB workflow 250202.181
  HFShowerLibrary::Params params;
  params.probMax_ = 1.;   // 0.5;
  params.backProb_ = 1.;  //0.2;
  params.dphi_ = 10 * deg;
  params.equalizeTimeShift_ = false;
  params.verbose_ = false;
  params.applyFidCut_ = true;

  SECTION("v4 file") {
    HFShowerLibrary::FileParams fileParams;
    edm::FileInPath p("SimG4CMS/Calo/data/HFShowerLibrary_npmt_noatt_eta4_16en_v4.root");
    fileParams.fileName_ = p.fullPath();
    fileParams.emBranchName_ = "emParticles";
    fileParams.hadBranchName_ = "hadParticles";
    fileParams.branchEvInfo_ = "";
    fileParams.fileVersion_ = 0;

    HFShowerLibrary showerLibrary(params, fileParams, std::move(fibreParams));
    SECTION("fillHits") {
      SetRandomEngine guard(11);

      SECTION("non EM or Hadron") {
        bool ok = false;
        auto hits = showerLibrary.fillHits({0, 0, 0}, {0, 0, 0}, 0, 0., ok, 1., 0.);
        REQUIRE(hits.empty());
        REQUIRE(not ok);
      }
      SECTION("photon within threshold") {
        bool ok = false;
        auto hits = showerLibrary.fillHits(
            {-470.637, -618.696, 11150}, {0.000467326, -0.00804975, 0.999967}, 22, 2.2 * GeV, ok, 1., 0.);
        REQUIRE(ok);
        REQUIRE(hits.size() == 2);
      }
    }
  }
  SECTION("v3 file") {
    HFShowerLibrary::FileParams fileParams;
    edm::FileInPath p("SimG4CMS/Calo/data/HFShowerLibrary_oldpmt_noatt_eta4_16en_v3.root");
    fileParams.fileName_ = p.fullPath();
    fileParams.emBranchName_ = "emParticles";
    fileParams.hadBranchName_ = "hadParticles";
    fileParams.branchEvInfo_ = "";
    fileParams.fileVersion_ = 0;

    HFShowerLibrary showerLibrary(params, fileParams, std::move(fibreParams));
    SECTION("fillHits") {
      SetRandomEngine guard(11);

      SECTION("non EM or Hadron") {
        bool ok = false;
        auto hits = showerLibrary.fillHits({0, 0, 0}, {0, 0, 0}, 0, 0., ok, 1., 0.);
        REQUIRE(hits.empty());
        REQUIRE(not ok);
      }
      SECTION("photon within threshold") {
        bool ok = false;
        auto hits = showerLibrary.fillHits(
            {-470.637, -618.696, 11150}, {0.000467326, -0.00804975, 0.999967}, 22, 5.2 * GeV, ok, 1., 0.);
        REQUIRE(ok);
        REQUIRE(hits.size() == 1);
      }
    }
  }
  SECTION("v6 file") {
    HFShowerLibrary::FileParams fileParams;
    edm::FileInPath p("SimG4CMS/Calo/data/HFShowerLibrary_run3_v6.root");
    fileParams.fileName_ = p.fullPath();
    fileParams.emBranchName_ = "emParticles";
    fileParams.hadBranchName_ = "hadParticles";
    fileParams.branchEvInfo_ = "";
    fileParams.fileVersion_ = 2;

    params.equalizeTimeShift_ = true;

    HFShowerLibrary showerLibrary(params, fileParams, std::move(fibreParams));
    SECTION("fillHits") {
      SetRandomEngine guard(11);

      SECTION("non EM or Hadron") {
        bool ok = false;
        auto hits = showerLibrary.fillHits({0, 0, 0}, {0, 0, 0}, 0, 0., ok, 1., 0.);
        REQUIRE(hits.empty());
        REQUIRE(not ok);
      }
      SECTION("photon within threshold") {
        bool ok = false;
        auto hits = showerLibrary.fillHits(
            {-470.637, -618.696, 11150}, {0.000467326, -0.00804975, 0.999967}, 22, 2.2 * GeV, ok, 1., 0.);
        REQUIRE(ok);
        REQUIRE(hits.size() == 5);
      }
    }
  }
}

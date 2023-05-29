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
    } catch (cms::Exception& e) {
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

bool operator==(HFShowerLibrary::Hit const& iLHS, HFShowerLibrary::Hit const& iRHS) {
  return iLHS.position == iRHS.position and iLHS.depth == iRHS.depth and iLHS.time == iRHS.time;
}

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
    fileParams.cacheBranches_ = false;

    HFShowerLibrary showerLibrary(params, fileParams, fibreParams);
    SECTION("fillHits") {
      SECTION("non EM or Hadron") {
        SetRandomEngine guard(11);
        bool ok = false;
        auto hits = showerLibrary.fillHits({0, 0, 0}, {0, 0, 0}, 0, 0., ok, 1., 0.);
        REQUIRE(hits.empty());
        REQUIRE(not ok);
      }
      SECTION("photon within threshold") {
        SetRandomEngine guard(11);
        bool ok = false;
        auto hits = showerLibrary.fillHits(
            {-470.637, -618.696, 11150}, {0.000467326, -0.00804975, 0.999967}, 22, 2.2 * GeV, ok, 1., 0.);
        REQUIRE(ok);
        REQUIRE(hits.size() == 2);
      }
      SECTION("pion within threshold") {
        SetRandomEngine guard(11);
        bool ok = false;
        auto hits = showerLibrary.fillHits(
            {-470.637, -618.696, 11150}, {0.000467326, -0.00804975, 0.999967}, 211, 2.2 * GeV, ok, 1., 0.);
        REQUIRE(ok);
        REQUIRE(hits.size() == 2);
      }
    }
    SECTION("em cache") {
      std::vector<HFShowerLibrary::Hit> nonCached;
      {
        SetRandomEngine guard(11);
        bool ok = false;
        nonCached = showerLibrary.fillHits(
            {-470.637, -618.696, 11150}, {0.000467326, -0.00804975, 0.999967}, 22, 2.2 * GeV, ok, 1., 0.);
      }
      std::vector<HFShowerLibrary::Hit> cached;
      {
        auto newFileParams = fileParams;
        newFileParams.cacheBranches_ = true;
        HFShowerLibrary showerLibrary(params, newFileParams, fibreParams);

        SetRandomEngine guard(11);
        bool ok = false;
        cached = showerLibrary.fillHits(
            {-470.637, -618.696, 11150}, {0.000467326, -0.00804975, 0.999967}, 22, 2.2 * GeV, ok, 1., 0.);
      }
      REQUIRE(nonCached == cached);
    }

    SECTION("had cache") {
      std::vector<HFShowerLibrary::Hit> nonCached;
      {
        SetRandomEngine guard(11);
        bool ok = false;
        nonCached = showerLibrary.fillHits(
            {-470.637, -618.696, 11150}, {0.000467326, -0.00804975, 0.999967}, 211, 2.2 * GeV, ok, 1., 0.);
      }
      std::vector<HFShowerLibrary::Hit> cached;
      {
        auto newFileParams = fileParams;
        newFileParams.cacheBranches_ = true;
        HFShowerLibrary showerLibrary(params, newFileParams, fibreParams);

        SetRandomEngine guard(11);
        bool ok = false;
        cached = showerLibrary.fillHits(
            {-470.637, -618.696, 11150}, {0.000467326, -0.00804975, 0.999967}, 211, 2.2 * GeV, ok, 1., 0.);
      }
      REQUIRE(nonCached == cached);
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
    fileParams.cacheBranches_ = false;

    HFShowerLibrary showerLibrary(params, fileParams, std::move(fibreParams));
    SECTION("fillHits") {
      SECTION("non EM or Hadron") {
        SetRandomEngine guard(11);
        bool ok = false;
        auto hits = showerLibrary.fillHits({0, 0, 0}, {0, 0, 0}, 0, 0., ok, 1., 0.);
        REQUIRE(hits.empty());
        REQUIRE(not ok);
      }
      SECTION("photon within threshold") {
        SetRandomEngine guard(11);
        bool ok = false;
        auto hits = showerLibrary.fillHits(
            {-470.637, -618.696, 11150}, {0.000467326, -0.00804975, 0.999967}, 22, 5.2 * GeV, ok, 1., 0.);
        REQUIRE(ok);
        REQUIRE(hits.size() == 1);
      }
      SECTION("pion within threshold") {
        SetRandomEngine guard(11);
        bool ok = false;
        auto hits = showerLibrary.fillHits(
            {-470.637, -618.696, 11150}, {0.000467326, -0.00804975, 0.999967}, 211, 5.2 * GeV, ok, 1., 0.);
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
    fileParams.cacheBranches_ = false;

    params.equalizeTimeShift_ = true;

    HFShowerLibrary showerLibrary(params, fileParams, std::move(fibreParams));
    SECTION("fillHits") {
      SECTION("non EM or Hadron") {
        SetRandomEngine guard(11);
        bool ok = false;
        auto hits = showerLibrary.fillHits({0, 0, 0}, {0, 0, 0}, 0, 0., ok, 1., 0.);
        REQUIRE(hits.empty());
        REQUIRE(not ok);
      }
      SECTION("photon within threshold") {
        SetRandomEngine guard(11);
        bool ok = false;
        auto hits = showerLibrary.fillHits(
            {-470.637, -618.696, 11150}, {0.000467326, -0.00804975, 0.999967}, 22, 2.2 * GeV, ok, 1., 0.);
        REQUIRE(ok);
        REQUIRE(hits.size() == 5);
      }
      SECTION("pion within threshold") {
        SetRandomEngine guard(11);
        bool ok = false;
        auto hits = showerLibrary.fillHits(
            {-470.637, -618.696, 11150}, {0.000467326, -0.00804975, 0.999967}, 211, 5.2 * GeV, ok, 1., 0.);
        REQUIRE(ok);
        REQUIRE(hits.size() == 2);
      }
    }
  }
}

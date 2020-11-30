#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Math/interface/GeantUnits.h"
#include "DetectorDescription/DDCMS/interface/DDDetector.h"
#include "DetectorDescription/DDCMS/interface/DDVectorRegistry.h"
#include "Geometry/Records/interface/DDVectorRegistryRcd.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/DDSpecParRegistryRcd.h"
#include "SimG4Core/DD4hepGeometry/interface/DD4hep_DDDWorld.h"
#include <DDG4/Geant4Converter.h>
#include <DD4hep/Detector.h>
#include <DD4hep/Handle.h>
#include <DD4hep/Filter.h>
#include <DD4hep/SpecParRegistry.h>
#include "G4LogicalVolume.hh"
#include "G4MTRunManagerKernel.hh"
#include "G4ProductionCuts.hh"
#include "G4RegionStore.hh"

#include <iostream>
#include <string>
#include <string_view>

using namespace std;
using namespace cms;
using namespace edm;
using namespace dd4hep;

#ifdef HAVE_GEANT4_UNITS
#define MM_2_CM 1.0
#else
#define MM_2_CM 0.1
#endif

namespace {
  bool sortByName(const std::pair<G4LogicalVolume*, const dd4hep::SpecPar*>& p1,
                  const std::pair<G4LogicalVolume*, const dd4hep::SpecPar*>& p2) {
    bool result = false;
    if (p1.first->GetName() > p2.first->GetName()) {
      result = true;
    }
    return result;
  }
}  // namespace

class DD4hepTestDDDWorld : public one::EDAnalyzer<> {
public:
  explicit DD4hepTestDDDWorld(const ParameterSet&);

  void beginJob() override {}
  void analyze(Event const& iEvent, EventSetup const&) override;
  void endJob() override {}

private:
  void update();
  void initialize(const dd4hep::sim::Geant4GeometryMaps::VolumeMap&);
  G4ProductionCuts* getProductionCuts(G4Region* region);

  const ESInputTag tag_;
  const dd4hep::SpecParRegistry* specPars_;
  G4MTRunManagerKernel* kernel_;
  dd4hep::SpecParRefs specs_;
  vector<pair<G4LogicalVolume*, const dd4hep::SpecPar*>> vec_;
  const string keywordRegion_;
  unique_ptr<DDDWorld> world_;
  int verbosity_;
};

DD4hepTestDDDWorld::DD4hepTestDDDWorld(const ParameterSet& iConfig)
    : tag_(iConfig.getParameter<ESInputTag>("DDDetector")), keywordRegion_("CMSCutsRegion") {
  verbosity_ = iConfig.getUntrackedParameter<int>("Verbosity", 1);
  kernel_ = new G4MTRunManagerKernel();
}

void DD4hepTestDDDWorld::analyze(const Event&, const EventSetup& iEventSetup) {
  LogVerbatim("Geometry") << "\nDD4hepTestDDDWorld::analyze: " << tag_;

  const DDVectorRegistryRcd& regRecord = iEventSetup.get<DDVectorRegistryRcd>();
  ESTransientHandle<DDVectorRegistry> reg;
  regRecord.get(tag_, reg);

  const auto& ddRecord = iEventSetup.get<IdealGeometryRecord>();
  ESTransientHandle<DDDetector> ddd;
  ddRecord.get(tag_, ddd);

  const DDSpecParRegistryRcd& specParRecord = iEventSetup.get<DDSpecParRegistryRcd>();
  ESTransientHandle<dd4hep::SpecParRegistry> registry;
  specParRecord.get(tag_, registry);
  specPars_ = registry.product();

  const dd4hep::Detector& detector = *ddd->description();
  dd4hep::sim::Geant4Converter g4Geo = dd4hep::sim::Geant4Converter(detector);
  g4Geo.debugMaterials = true;
  g4Geo.debugElements = true;
  g4Geo.debugShapes = true;
  g4Geo.debugVolumes = true;
  g4Geo.debugPlacements = true;
  g4Geo.create(detector.world());

  dd4hep::sim::Geant4GeometryMaps::VolumeMap lvMap;
  world_.reset(new DDDWorld(ddd.product(), lvMap));
  initialize(lvMap);
  update();
  LogVerbatim("Geometry") << "Done.";
}

void DD4hepTestDDDWorld::initialize(const dd4hep::sim::Geant4GeometryMaps::VolumeMap& vmap) {
  specPars_->filter(specs_, keywordRegion_);

  LogVerbatim("Geometry").log([&](auto& log) {
    for (auto const& it : vmap) {
      for (auto const& fit : specs_) {
        for (auto const& sit : fit.second->numpars) {
          log << sit.first << " =  " << sit.second[0] << "\n";
        }
        for (auto const& pit : fit.second->paths) {
          const std::string_view selection = dd4hep::dd::realTopName(pit);
          const std::string_view name = dd4hep::dd::noNamespace(it.first.name());
          log << selection << "\n";
          log << "   compare equal to " << name << " ... ";
          if (!(dd4hep::dd::isRegex(selection))
                  ? dd4hep::dd::compareEqual(name, selection)
                  : regex_match(name.begin(), name.end(), regex(std::string(selection)))) {
            vec_.emplace_back(std::make_pair<G4LogicalVolume*, const dd4hep::SpecPar*>(&*it.second, &*fit.second));
            log << "   are equal!\n";
          } else
            log << "   nope.\n";
        }
      }
    }
  });

  // sort all root volumes - to get the same sequence at every run of the application.
  sort(begin(vec_), end(vec_), &sortByName);

  // Now generate all the regions
  for (auto const& it : vec_) {
    auto regName = it.second->strValue(keywordRegion_);
    G4Region* region = G4RegionStore::GetInstance()->FindOrCreateRegion({regName.data(), regName.size()});
    region->AddRootLogicalVolume(it.first);
    LogVerbatim("Geometry") << it.first->GetName() << ": " << it.second->strValue(keywordRegion_);
    LogVerbatim("Geometry") << " MakeRegions: added " << it.first->GetName() << " to region " << region->GetName();
  }
}

void DD4hepTestDDDWorld::update() {
  LogVerbatim("Geometry").log([&](auto& log) {
    log << "DD4hepTestDDDWorld::update()\n";
    for (const auto& t : vec_) {
      log << t.first->GetName() << ":\n";
      for (const auto& kl : t.second->numpars) {
        log << kl.first << " = ";
        for (const auto& kil : kl.second) {
          log << kil << ", ";
        }
      }
      log << "\n";
    }
    log << "DD4hepTestDDDWorld::update() done!\n";
  });
  // Loop over all DDLP and provide the cuts for each region
  for (auto const& it : vec_) {
    auto regName = it.second->strValue(keywordRegion_);
    G4Region* region = G4RegionStore::GetInstance()->FindOrCreateRegion({regName.data(), regName.size()});

    //
    // search for production cuts
    // you must have four of them: e+ e- gamma proton
    //

    // Geant4 expects mm units. DD4hep returns cm, so must convert to mm.
    double gammacut = it.second->dblValue("ProdCutsForGamma") / MM_2_CM;
    double electroncut = it.second->dblValue("ProdCutsForElectrons") / MM_2_CM;
    double positroncut = it.second->dblValue("ProdCutsForPositrons") / MM_2_CM;
    double protoncut = it.second->dblValue("ProdCutsForProtons") / MM_2_CM;
    if (protoncut == 0) {
      protoncut = electroncut;
    }

    //
    // For the moment I assume all of the four are set
    //
    G4ProductionCuts* prodCuts = getProductionCuts(region);
    prodCuts->SetProductionCut(gammacut, idxG4GammaCut);
    prodCuts->SetProductionCut(electroncut, idxG4ElectronCut);
    prodCuts->SetProductionCut(positroncut, idxG4PositronCut);
    prodCuts->SetProductionCut(protoncut, idxG4ProtonCut);
    if (verbosity_ > 0) {
      LogVerbatim("Geometry") << "DDG4ProductionCuts : Setting cuts for " << regName
                              << "\n    Electrons: " << electroncut << " mm\n    Positrons: " << positroncut
                              << " mm\n    Gamma    : " << gammacut << " mm\n    Protons  : " << protoncut << " mm\n";
    }
  }
}

G4ProductionCuts* DD4hepTestDDDWorld::getProductionCuts(G4Region* region) {
  G4ProductionCuts* prodCuts = region->GetProductionCuts();
  if (!prodCuts) {
    prodCuts = new G4ProductionCuts();
    region->SetProductionCuts(prodCuts);
  }
  return prodCuts;
}

DEFINE_FWK_MODULE(DD4hepTestDDDWorld);

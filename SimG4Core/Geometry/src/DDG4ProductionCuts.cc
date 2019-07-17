#include "SimG4Core/Geometry/interface/DDG4ProductionCuts.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "G4ProductionCuts.hh"
#include "G4RegionStore.hh"
#include "G4Region.hh"
#include "G4LogicalVolume.hh"

#include <algorithm>

DDG4ProductionCuts::DDG4ProductionCuts(const G4LogicalVolumeToDDLogicalPartMap& map,
                                       int verb, bool pcut)
  : map_(map), keywordRegion_("CMSCutsRegion"), verbosity_(verb), protonCut_(pcut) {
  initialize();
}

DDG4ProductionCuts::~DDG4ProductionCuts() {}

/** helper function to compare parts through their name instead of comparing them
    by their pointers. 
    It's guaranteed to produce the same order in subsequent application runs,
    while pointers usually can't guarantee this
*/
bool dd_is_greater(const std::pair<G4LogicalVolume*, DDLogicalPart>& p1,
                   const std::pair<G4LogicalVolume*, DDLogicalPart>& p2) {
  bool result = false;
  if (p1.second.name().ns() > p2.second.name().ns()) {
    result = true;
  }
  if (p1.second.name().ns() == p2.second.name().ns()) {
    if (p1.second.name().name() > p2.second.name().name()) {
      result = true;
    }
    if (p1.second.name().name() == p2.second.name().name()) {
      if (p1.first->GetName() > p2.first->GetName()) {
        result = true;
      }
    }
  }
  return result;
}

void DDG4ProductionCuts::initialize() {
  vec_ = map_.all(keywordRegion_);
  // sort all root volumes - to get the same sequence at every run of the application.
  // (otherwise, the sequence will depend on the pointer (memory address) of the
  // involved objects, because 'new' does no guarantee that you allways get a
  // higher (or lower) address when allocating an object of the same type ...
  sort(vec_.begin(), vec_.end(), &dd_is_greater);
  if (verbosity_ > -1) {
    edm::LogVerbatim("Geometry") << " DDG4ProductionCuts : got " << vec_.size() << " region roots.\n"
                                 << " DDG4ProductionCuts : List of all roots:";
    for (auto const& vv : vec_)
      edm::LogVerbatim("Geometry") << "    " << vv.first->GetName() << " : " << vv.second.name();
  }

  // Now generate all the regions
  std::string curName = "";
  std::string regionName = "";
  G4Region* region = nullptr;
  G4RegionStore* store = G4RegionStore::GetInstance();
  for (auto const& vv : vec_) {
    unsigned int num = map_.toString(keywordRegion_, vv.second, regionName);
    edm::LogVerbatim("Geometry") << "  num  " << num << " regionName: " << regionName << " " << store;

    if (num != 1) {
      throw cms::Exception("SimG4CorePhysics", " DDG4ProductionCuts::initialize: Problem with Region tags.");
    }
    if (regionName != curName) {
      edm::LogVerbatim("Geometry") << "DDG4ProductionCuts : regionName " << regionName << " " << store;
      region = store->FindOrCreateRegion(regionName);
      edm::LogVerbatim("Geometry") << "DDG4ProductionCuts : region " << region;
      if (!region) {
        throw cms::Exception("SimG4CoreGeometry", " DDG4ProductionCuts::initialize: Problem with Region tags.");
      }
      curName = regionName;
      edm::LogVerbatim("Geometry") << "DDG4ProductionCuts : new G4Region " << vv.first->GetName();
      setProdCuts(vv.second, region);
    }

    region->AddRootLogicalVolume(vv.first);

    if (verbosity_ > -1)
      edm::LogVerbatim("Geometry") << "  added " << vv.first->GetName() << " to region " << region->GetName();
  }
}

void DDG4ProductionCuts::setProdCuts(const DDLogicalPart lpart, G4Region* region) {
  //
  // search for production cuts
  // you must have four of them: e+ e- gamma proton
  //
  double gammacut = 0.0;
  double electroncut = 0.0;
  double positroncut = 0.0;
  double protoncut = 0.0;
  int temp = map_.toDouble("ProdCutsForGamma", lpart, gammacut);
  if (temp != 1) {
    throw cms::Exception(
        "SimG4CorePhysics",
        " DDG4ProductionCuts::setProdCuts: Problem with Region tags - no/more than one ProdCutsForGamma.");
  }
  temp = map_.toDouble("ProdCutsForElectrons", lpart, electroncut);
  if (temp != 1) {
    throw cms::Exception(
        "SimG4CorePhysics",
        " DDG4ProductionCuts::setProdCuts: Problem with Region tags - no/more than one ProdCutsForElectrons.");
  }
  temp = map_.toDouble("ProdCutsForPositrons", lpart, positroncut);
  if (temp != 1) {
    throw cms::Exception(
        "SimG4CorePhysics",
        " DDG4ProductionCuts::setProdCuts: Problem with Region tags - no/more than one ProdCutsForPositrons.");
  }
  temp = map_.toDouble("ProdCutsForProtons", lpart, protoncut);
  if (temp == 0) {
    // There is no ProdCutsForProtons set in XML,
    // check if it's a legacy geometry scenario without it
    if (protonCut_) {
      protoncut = electroncut;
    } else {
      protoncut = 0.;
    }
  } else if (temp != 1) {
    throw cms::Exception(
        "SimG4CorePhysics",
        " DDG4ProductionCuts::setProdCuts: Problem with Region tags - more than one ProdCutsForProtons.");
  }

  //
  // Create and fill production cuts
  //
  G4ProductionCuts* prodCuts = region->GetProductionCuts();
  if (!prodCuts) {
    prodCuts = new G4ProductionCuts();
    region->SetProductionCuts(prodCuts);
  }
  prodCuts->SetProductionCut(gammacut, idxG4GammaCut);
  prodCuts->SetProductionCut(electroncut, idxG4ElectronCut);
  prodCuts->SetProductionCut(positroncut, idxG4PositronCut);
  prodCuts->SetProductionCut(protoncut, idxG4ProtonCut);
  if (verbosity_ > -1) {
    edm::LogVerbatim("Geometry") << "DDG4ProductionCuts : Setting cuts for " << region->GetName()
                                 << "\n    Electrons: " << electroncut << "\n    Positrons: " << positroncut
                                 << "\n    Gamma    : " << gammacut << "\n    Proton   : " << protoncut;
  }
}

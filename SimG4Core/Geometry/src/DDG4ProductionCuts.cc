#include "SimG4Core/Geometry/interface/DDG4ProductionCuts.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <DD4hep/Filter.h>

#include "G4ProductionCuts.hh"
#include "G4RegionStore.hh"
#include "G4Region.hh"
#include "G4LogicalVolume.hh"

#include <algorithm>

namespace {
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

  bool sortByName(const std::pair<G4LogicalVolume*, const dd4hep::SpecPar*>& p1,
                  const std::pair<G4LogicalVolume*, const dd4hep::SpecPar*>& p2) {
    bool result = false;
    if (p1.first->GetName() > p2.first->GetName()) {
      result = true;
    }
    return result;
  }
}  // namespace

DDG4ProductionCuts::DDG4ProductionCuts(const G4LogicalVolumeToDDLogicalPartMap* map, int verb, bool pcut)
    : map_(map), keywordRegion_("CMSCutsRegion"), verbosity_(verb), protonCut_(pcut) {
  initialize();
}

DDG4ProductionCuts::DDG4ProductionCuts(const dd4hep::SpecParRegistry* specPars,
                                       const dd4hep::sim::Geant4GeometryMaps::VolumeMap* map,
                                       int verb,
                                       bool pcut)
    : dd4hepMap_(map), specPars_(specPars), keywordRegion_("CMSCutsRegion"), verbosity_(verb), protonCut_(pcut) {
  dd4hepInitialize();
}

void DDG4ProductionCuts::initialize() {
  vec_ = map_->all(keywordRegion_);
  // sort all root volumes - to get the same sequence at every run of the application.
  // (otherwise, the sequence will depend on the pointer (memory address) of the
  // involved objects, because 'new' does no guarantee that you allways get a
  // higher (or lower) address when allocating an object of the same type ...
  sort(vec_.begin(), vec_.end(), &dd_is_greater);
  if (verbosity_ > 0) {
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
    unsigned int num = map_->toString(keywordRegion_, vv.second, regionName);
    edm::LogVerbatim("Geometry") << "  num  " << num << " regionName: " << regionName << ", the store of size "
                                 << store->size();
    if (num != 1) {
      throw cms::Exception("SimG4CoreGeometry", " DDG4ProductionCuts::initialize: Problem with Region tags.");
    }
    if (regionName != curName) {
      edm::LogVerbatim("Geometry") << "DDG4ProductionCuts : regionName " << regionName << ", the store of size "
                                   << store->size();
      region = store->FindOrCreateRegion(regionName);
      edm::LogVerbatim("Geometry") << "DDG4ProductionCuts : region " << region->GetName();
      if (!region) {
        throw cms::Exception("SimG4CoreGeometry", " DDG4ProductionCuts::initialize: Problem with Region tags.");
      }
      curName = regionName;
      edm::LogVerbatim("Geometry") << "DDG4ProductionCuts : new G4Region " << vv.first->GetName();
      setProdCuts(vv.second, region);
    }

    region->AddRootLogicalVolume(vv.first);

    if (verbosity_ > 0)
      edm::LogVerbatim("Geometry") << "  added " << vv.first->GetName() << " to region " << region->GetName();
  }
}

void DDG4ProductionCuts::dd4hepInitialize() {
  dd4hep::SpecParRefs specs;
  specPars_->filter(specs, keywordRegion_);

  for (auto const& it : *dd4hepMap_) {
    for (auto const& fit : specs) {
      for (auto const& pit : fit.second->paths) {
        if (dd4hep::dd::compareEqual(dd4hep::dd::noNamespace(it.first.name()), dd4hep::dd::realTopName(pit))) {
          dd4hepVec_.emplace_back(std::make_pair<G4LogicalVolume*, const dd4hep::SpecPar*>(&*it.second, &*fit.second));
        }
      }
    }
  }
  // sort all root volumes - to get the same sequence at every run of the application.
  sort(begin(dd4hepVec_), end(dd4hepVec_), &sortByName);

  // Now generate all the regions
  for (auto const& it : dd4hepVec_) {
    auto regName = it.second->strValue(keywordRegion_);
    G4Region* region = G4RegionStore::GetInstance()->FindOrCreateRegion({regName.data(), regName.size()});
    region->AddRootLogicalVolume(it.first);
    edm::LogVerbatim("Geometry") << it.first->GetName() << ": " << it.second->strValue(keywordRegion_);
    edm::LogVerbatim("Geometry") << " MakeRegions: added " << it.first->GetName() << " to region " << region->GetName();
    edm::LogVerbatim("Geometry").log([&](auto& log) {
      for (auto const& sit : it.second->spars) {
        log << sit.first << " =  " << sit.second[0] << "\n";
      }
    });
    setProdCuts(it.second, region);
  }

  if (verbosity_ > 0) {
    edm::LogVerbatim("SimG4CoreGeometry") << " DDG4ProductionCuts (New) : starting\n"
                                          << " DDG4ProductionCuts : Got " << dd4hepVec_.size() << " region roots.\n"
                                          << " DDG4ProductionCuts : List of all roots:";
    for (size_t jj = 0; jj < dd4hepVec_.size(); ++jj)
      edm::LogVerbatim("SimG4CoreGeometry")
          << "   DDG4ProductionCuts : root=" << dd4hepVec_[jj].first << " , " << dd4hepVec_[jj].second;
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
  int temp = map_->toDouble("ProdCutsForGamma", lpart, gammacut);
  if (temp != 1) {
    throw cms::Exception(
        "SimG4CorePhysics",
        " DDG4ProductionCuts::setProdCuts: Problem with Region tags - no/more than one ProdCutsForGamma.");
  }
  temp = map_->toDouble("ProdCutsForElectrons", lpart, electroncut);
  if (temp != 1) {
    throw cms::Exception(
        "SimG4CorePhysics",
        " DDG4ProductionCuts::setProdCuts: Problem with Region tags - no/more than one ProdCutsForElectrons.");
  }
  temp = map_->toDouble("ProdCutsForPositrons", lpart, positroncut);
  if (temp != 1) {
    throw cms::Exception(
        "SimG4CorePhysics",
        " DDG4ProductionCuts::setProdCuts: Problem with Region tags - no/more than one ProdCutsForPositrons.");
  }
  temp = map_->toDouble("ProdCutsForProtons", lpart, protoncut);
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
  if (verbosity_ > 0) {
    edm::LogVerbatim("Geometry") << "DDG4ProductionCuts : Setting cuts for " << region->GetName()
                                 << "\n    Electrons: " << electroncut << "\n    Positrons: " << positroncut
                                 << "\n    Gamma    : " << gammacut << "\n    Proton   : " << protoncut;
  }
}

void DDG4ProductionCuts::setProdCuts(const dd4hep::SpecPar* spec, G4Region* region) {
  //
  // Create and fill production cuts
  //
  G4ProductionCuts* prodCuts = region->GetProductionCuts();
  if (!prodCuts) {
    // FIXME: Here we use a dd4hep string to double evaluator
    //        Beware of the units!!!

    //
    // search for production cuts
    // you must have four of them: e+ e- gamma proton
    //
    double gammacut = 0.0;
    double electroncut = 0.0;
    double positroncut = 0.0;
    double protoncut = 0.0;

    auto gammacutStr = spec->strValue("ProdCutsForGamma");
    if (gammacutStr.empty()) {
      throw cms::Exception(
          "SimG4CorePhysics",
          " DDG4ProductionCuts::setProdCuts: Problem with Region tags - no/more than one ProdCutsForGamma.");
    }
    gammacut = dd4hep::_toDouble({gammacutStr.data(), gammacutStr.size()});

    auto electroncutStr = spec->strValue("ProdCutsForElectrons");
    if (electroncutStr.empty()) {
      throw cms::Exception(
          "SimG4CorePhysics",
          " DDG4ProductionCuts::setProdCuts: Problem with Region tags - no/more than one ProdCutsForElectrons.");
    }
    electroncut = dd4hep::_toDouble({electroncutStr.data(), electroncutStr.size()});

    auto positroncutStr = spec->strValue("ProdCutsForPositrons");
    if (positroncutStr.empty()) {
      throw cms::Exception(
          "SimG4CorePhysics",
          " DDG4ProductionCuts::setProdCuts: Problem with Region tags - no/more than one ProdCutsForPositrons.");
    }
    positroncut = dd4hep::_toDouble({positroncutStr.data(), positroncutStr.size()});

    if (!spec->hasValue("ProdCutsForProtons")) {
      // There is no ProdCutsForProtons set in XML,
      // check if it's a legacy geometry scenario without it
      if (protonCut_) {
        protoncut = electroncut;
      } else {
        protoncut = 0.;
      }
    } else {
      auto protoncutStr = spec->strValue("ProdCutsForProtons");
      if (protoncutStr.empty()) {
        throw cms::Exception(
            "SimG4CorePhysics",
            " DDG4ProductionCuts::setProdCuts: Problem with Region tags - more than one ProdCutsForProtons.");
      }
      protoncut = dd4hep::_toDouble({protoncutStr.data(), protoncutStr.size()});
    }

    prodCuts = new G4ProductionCuts();
    region->SetProductionCuts(prodCuts);

    prodCuts->SetProductionCut(gammacut, idxG4GammaCut);
    prodCuts->SetProductionCut(electroncut, idxG4ElectronCut);
    prodCuts->SetProductionCut(positroncut, idxG4PositronCut);
    prodCuts->SetProductionCut(protoncut, idxG4ProtonCut);
    if (verbosity_ > 0) {
      edm::LogVerbatim("Geometry") << "DDG4ProductionCuts : Setting cuts for " << region->GetName()
                                   << "\n    Electrons: " << electroncut << "\n    Positrons: " << positroncut
                                   << "\n    Gamma    : " << gammacut << "\n    Proton   : " << protoncut;
    }
  } else {
    if (verbosity_ > 0) {
      edm::LogVerbatim("Geometry")
          << "DDG4ProductionCuts : Cuts are already set for " << region->GetName()
          << "\n    Electrons: " << region->GetProductionCuts()->GetProductionCut(idxG4ElectronCut)
          << "\n    Positrons: " << region->GetProductionCuts()->GetProductionCut(idxG4PositronCut)
          << "\n    Gamma    : " << region->GetProductionCuts()->GetProductionCut(idxG4GammaCut)
          << "\n    Proton   : " << region->GetProductionCuts()->GetProductionCut(idxG4ProtonCut);
    }
  }
}

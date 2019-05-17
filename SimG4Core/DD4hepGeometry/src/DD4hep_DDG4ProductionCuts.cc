#include "SimG4Core/DD4hepGeometry/interface/DD4hep_DDG4ProductionCuts.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DetectorDescription/DDCMS/interface/Filter.h"

#include "G4ProductionCuts.hh"
#include "G4Region.hh"
#include "G4RegionStore.hh"
#include "G4LogicalVolume.hh"

#include <algorithm>
#include <string_view>
#include <utility>

DD4hep_DDG4ProductionCuts::DD4hep_DDG4ProductionCuts(const cms::DDSpecParRegistry* specPars,
						     const dd4hep::sim::Geant4GeometryMaps::VolumeMap& lvMap,
						     int verb, 
						     const edm::ParameterSet & p) 
  : m_specPars(specPars), m_map(lvMap), m_verbosity(verb) {
  m_keywordRegion = "CMSCutsRegion";
  m_protonCut = p.getUntrackedParameter<bool>("CutsOnProton",true);  
  initialize();
}

DD4hep_DDG4ProductionCuts::~DD4hep_DDG4ProductionCuts(){
}

void DD4hep_DDG4ProductionCuts::update() {
  //
  // Loop over all DDLP and provide the cuts for each region
  //
  std::cout << "DD4hep_DDG4ProductionCuts::update()\n";
  for(const auto& t: m_vec) {
    std::cout << t.first->GetName() << ":\n";
    for(const auto& kl : t.second->spars) {
      std::cout << kl.first << " = "; 
      for(const auto& kil : kl.second) {
	std::cout << kil << ", ";
      }
    }
    std::cout << "\n";
  }
  std::cout << "DD4hep_DDG4ProductionCuts::update() done!\n";

  // for (G4LogicalVolumeToDDLogicalPartMap::Vector::iterator tit = vec_.begin();
  //      tit != vec_.end(); tit++){
  //   setProdCuts((*tit).second,(*tit).first);
  // }
}

namespace {
  std::string_view noNamespace(std::string_view input) {
    std::string_view v = input;
    auto first = v.find_first_of(":");
    v.remove_prefix(std::min(first+1, v.size()));
    return v;
  }
}

void DD4hep_DDG4ProductionCuts::initialize() {

  m_specPars->filter(m_specs, m_keywordRegion);

  for(auto const& it : m_map) {
    for(auto const& fit : m_specs) {
      for(auto const& sit : fit->spars) {
	std::cout << sit.first << " =  " << sit.second[0] << "\n";
      }
      for(auto const& pit : fit->paths) {
	std::cout << cms::dd::realTopName(pit) << "\n";
	std::cout << "   compare equal to " << noNamespace(it.first.name()) << " ... ";
	if(cms::dd::compareEqual(noNamespace(it.first.name()), cms::dd::realTopName(pit))) {
	  m_vec.emplace_back(std::make_pair<G4LogicalVolume*, const cms::DDSpecPar*>(&*it.second, &*fit));
	  std::cout << "   are equal!\n";
	} else
	  std::cout << "   nope.\n";	
      }
    }
  }
  // sort all root volumes - to get the same sequence at every run of the application.
  // (otherwise, the sequence will depend on the pointer (memory address) of the 
  // involved objects, because 'new' does no guarantee that you allways get a
  // higher (or lower) address when allocating an object of the same type ...
  ////// sort(vec_.begin(),vec_.end(),&dd_is_greater);
  if ( m_verbosity > 0 ) {
    LogDebug("Physics") <<" DDG4ProductionCuts (New) : starting\n"
			<<" DDG4ProductionCuts : Got "<< m_vec.size()
			<<" region roots.\n"
			<<" DDG4ProductionCuts : List of all roots:";
    for ( size_t jj=0; jj < m_vec.size(); ++jj)
      LogDebug("Physics") << "   DDG4ProductionCuts : root=" 
			  << m_vec[jj];
  }

  // Now generate all the regions
  // for (G4LogicalVolumeToDDLogicalPartMap::Vector::iterator tit = vec_.begin();
  //      tit != vec_.end(); tit++) {

  //   std::string  regionName;
  //   unsigned int num= map_.toString(m_KeywordRegion,(*tit).second,regionName);
  
  //   if (num != 1) {
  //     throw cms::Exception("SimG4CorePhysics", " DDG4ProductionCuts::initialize: Problem with Region tags.");
  //   }
  //   G4Region * region = getRegion(regionName);
  //   region->AddRootLogicalVolume((*tit).first);
  
  //   if ( m_Verbosity > 0 )
  //     LogDebug("Physics") << " MakeRegions: added " <<((*tit).first)->GetName()
  // 			  << " to region " << region->GetName();
  // }
}

void DD4hep_DDG4ProductionCuts::setProdCuts( G4LogicalVolume* lvol ) {  
  
  if ( m_verbosity > 0 ) 
    LogDebug("Physics") <<" DDG4ProductionCuts: inside setProdCuts";
  
  // G4Region * region = nullptr;
  
  // std::string  regionName;
  // unsigned int num= map_.toString(m_KeywordRegion,lpart,regionName);
  
  // if (num != 1) {
  //   throw cms::Exception("SimG4CorePhysics", " DDG4ProductionCuts::setProdCuts: Problem with Region tags.");
  // }
  // if ( m_Verbosity > 0 ) LogDebug("Physics") << "Using region " << regionName;

  // region = getRegion(regionName);

  // //
  // // search for production cuts
  // // you must have four of them: e+ e- gamma proton
  // //
  // double gammacut;
  // double electroncut;
  // double positroncut;
  // double protoncut = 0.0;
  // int temp =  map_.toDouble("ProdCutsForGamma",lpart,gammacut);
  // if (temp != 1){
  //   throw cms::Exception("SimG4CorePhysics", " DDG4ProductionCuts::setProdCuts: Problem with Region tags - no/more than one ProdCutsForGamma.");
  // }
  // temp =  map_.toDouble("ProdCutsForElectrons",lpart,electroncut);
  // if (temp != 1){
  //   throw cms::Exception("SimG4CorePhysics", " DDG4ProductionCuts::setProdCuts: Problem with Region tags - no/more than one ProdCutsForElectrons.");
  // }
  // temp =  map_.toDouble("ProdCutsForPositrons",lpart,positroncut);
  // if (temp != 1) {
  //   throw cms::Exception("SimG4CorePhysics", " DDG4ProductionCuts::setProdCuts: Problem with Region tags - no/more than one ProdCutsForPositrons.");
  // }
  // //
  // // For the moment I assume all of the three are set
  // //
  // G4ProductionCuts * prodCuts = getProductionCuts(region);
  // prodCuts->SetProductionCut( gammacut, idxG4GammaCut );
  // prodCuts->SetProductionCut( electroncut, idxG4ElectronCut );
  // prodCuts->SetProductionCut( positroncut, idxG4PositronCut );
  // // For recoil use the same cut as for e-
  // if(m_protonCut) { protoncut = electroncut; }
  // prodCuts->SetProductionCut( protoncut, idxG4ProtonCut );
  // if ( m_Verbosity > 0 ) {
  //   LogDebug("Physics") << "DDG4ProductionCuts : Setting cuts for " 
  // 			<< regionName << "\n    Electrons: " << electroncut
  // 			<< "\n    Positrons: " << positroncut
  // 			<< "\n    Gamma    : " << gammacut;
  // }
}

G4Region * DD4hep_DDG4ProductionCuts::getRegion(const std::string & regName) {
  G4Region * reg =  G4RegionStore::GetInstance()->FindOrCreateRegion (regName);
  return reg;
}

G4ProductionCuts * DD4hep_DDG4ProductionCuts::getProductionCuts( G4Region* reg ) {

  G4ProductionCuts * prodCuts = reg->GetProductionCuts();
  if( !prodCuts ) {
    prodCuts = new G4ProductionCuts();
    reg->SetProductionCuts(prodCuts);
  }
  return prodCuts;
}

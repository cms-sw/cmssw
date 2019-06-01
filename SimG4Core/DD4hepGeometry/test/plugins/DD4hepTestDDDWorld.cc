#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/DDCMS/interface/DDDetector.h"
#include "DetectorDescription/DDCMS/interface/DDVectorRegistry.h"
#include "DetectorDescription/DDCMS/interface/DDSpecParRegistry.h"
#include "DetectorDescription/DDCMS/interface/Filter.h"
#include "Geometry/Records/interface/DDVectorRegistryRcd.h"
#include "Geometry/Records/interface/GeometryFileRcd.h"
#include "Geometry/Records/interface/DDSpecParRegistryRcd.h"
#include "SimG4Core/DD4hepGeometry/interface/DD4hep_DDDWorld.h"
#include "DDG4/Geant4Converter.h"
#include "DD4hep/Detector.h"
#include "G4LogicalVolume.hh"

#include <iostream>
#include <string>
#include <string_view>

using namespace std;
using namespace cms;
using namespace edm;
using namespace dd4hep;

namespace {
  string_view noNamespace(string_view input) {
    string_view v = input;
    auto first = v.find_first_of(":");
    v.remove_prefix(min(first+1, v.size()));
    return v;
  }
}

class DD4hepTestDDDWorld : public one::EDAnalyzer<> {
public:
  explicit DD4hepTestDDDWorld(const ParameterSet&);

  void beginJob() override {}
  void analyze(Event const& iEvent, EventSetup const&) override;
  void endJob() override {}

private:
  void update();
  void initialize(const dd4hep::sim::Geant4GeometryMaps::VolumeMap&);
  
  const ESInputTag m_tag;
  const DDSpecParRegistry* m_specPars;
  DDSpecParRefs m_specs;
  vector<pair<G4LogicalVolume*, const DDSpecPar*>> m_vec;
  string_view m_keywordRegion;
  unique_ptr<DDDWorld> m_world;
};

DD4hepTestDDDWorld::DD4hepTestDDDWorld(const ParameterSet& iConfig)
  : m_tag(iConfig.getParameter<ESInputTag>("DDDetector"))
{
  m_keywordRegion = "CMSCutsRegion";
}

void
DD4hepTestDDDWorld::analyze(const Event&, const EventSetup& iEventSetup)
{
  LogVerbatim("Geometry") << "\nDD4hepTestDDDWorld::analyze: " << m_tag;

  const DDVectorRegistryRcd& regRecord = iEventSetup.get<DDVectorRegistryRcd>();
  ESTransientHandle<DDVectorRegistry> reg;
  regRecord.get(m_tag, reg);

  const GeometryFileRcd& ddRecord = iEventSetup.get<GeometryFileRcd>();
  ESTransientHandle<DDDetector> ddd;
  ddRecord.get(m_tag, ddd);

  const DDSpecParRegistryRcd& specParRecord = iEventSetup.get<DDSpecParRegistryRcd>();
  ESTransientHandle<DDSpecParRegistry> registry;
  specParRecord.get(m_tag, registry);
  m_specPars = registry.product();
    
  const dd4hep::Detector& detector = *ddd->description();
  dd4hep::sim::Geant4Converter g4Geo = dd4hep::sim::Geant4Converter(detector);
  g4Geo.debugMaterials = true;
  g4Geo.debugElements = true;
  g4Geo.debugShapes = true;
  g4Geo.debugVolumes = true;
  g4Geo.debugPlacements = true;
  g4Geo.create(detector.world());
  
  dd4hep::sim::Geant4GeometryMaps::VolumeMap lvMap;
  m_world.reset(new DDDWorld(ddd.product(), lvMap));
  initialize(lvMap);
  update();
  LogVerbatim("Geometry") << "Done.";
}

void DD4hepTestDDDWorld::initialize(const dd4hep::sim::Geant4GeometryMaps::VolumeMap& vmap) {
  // FIXME: when PR#26890 is in IBs
  m_specPars->filter(m_specs, m_keywordRegion, "");

  LogVerbatim("Geometry").log([&](auto& log) {
    for(auto const& it : vmap) {
      for(auto const& fit : m_specs) {
        for(auto const& sit : fit->spars) {
	  log << sit.first << " =  " << sit.second[0] << "\n";
	}
	for(auto const& pit : fit->paths) {
	  log << cms::dd::realTopName(pit) << "\n";
	  log << "   compare equal to " << noNamespace(it.first.name()) << " ... ";
	  if(cms::dd::compareEqual(noNamespace(it.first.name()), cms::dd::realTopName(pit))) {
	    m_vec.emplace_back(std::make_pair<G4LogicalVolume*, const cms::DDSpecPar*>(&*it.second, &*fit));
	    log << "   are equal!\n";
	  } else
	    log << "   nope.\n";	
        }
      }
    }
  });
}

void DD4hepTestDDDWorld::update() {
  LogVerbatim("Geometry").log([&](auto& log) {
    log << "DD4hepTestDDDWorld::update()";
    for(const auto& t: m_vec) {
      log << t.first->GetName() << ":\n";
      for(const auto& kl : t.second->spars) {
        log << kl.first << " = "; 
        for(const auto& kil : kl.second) {
	  log << kil << ", ";
        }
      }
      log << "\n";
    }
    log << "DD4hepTestDDDWorld::update() done!\n";
  });
}

DEFINE_FWK_MODULE(DD4hepTestDDDWorld);








  

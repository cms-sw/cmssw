#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/DDCMS/interface/DDDetector.h"
#include "DetectorDescription/DDCMS/interface/DDVectorRegistry.h"
#include "Geometry/Records/interface/DDVectorRegistryRcd.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DDG4/Geant4Converter.h"
#include "DD4hep/Detector.h"

#include <iostream>
#include <string>

using namespace std;
using namespace cms;
using namespace edm;
using namespace dd4hep;

class DD4hepTestG4Geometry : public one::EDAnalyzer<> {
public:
  explicit DD4hepTestG4Geometry(const ParameterSet&);

  void beginJob() override {}
  void analyze(Event const& iEvent, EventSetup const&) override;
  void endJob() override {}

private:
  const ESInputTag m_tag;
  const string m_detElementPath;
  const string m_placedVolPath;
};

DD4hepTestG4Geometry::DD4hepTestG4Geometry(const ParameterSet& iConfig)
    : m_tag(iConfig.getParameter<ESInputTag>("DDDetector")) {}

void DD4hepTestG4Geometry::analyze(const Event&, const EventSetup& iEventSetup) {
  LogVerbatim("Geometry") << "\nDD4hepTestG4Geometry::analyze: " << m_tag;

  const DDVectorRegistryRcd& regRecord = iEventSetup.get<DDVectorRegistryRcd>();
  ESTransientHandle<DDVectorRegistry> reg;
  regRecord.get(m_tag.module(), reg);

  const auto& ddRecord = iEventSetup.get<IdealGeometryRecord>();
  ESTransientHandle<DDDetector> ddd;
  ddRecord.get(m_tag.module(), ddd);

  const dd4hep::Detector& detector = *ddd->description();
  dd4hep::sim::Geant4Converter g4Geo = dd4hep::sim::Geant4Converter(detector);
  g4Geo.debugMaterials = true;
  g4Geo.debugElements = true;
  g4Geo.debugShapes = true;
  g4Geo.debugVolumes = true;
  g4Geo.debugPlacements = true;
  g4Geo.create(detector.world());

  LogVerbatim("Geometry") << "Done.";
}

DEFINE_FWK_MODULE(DD4hepTestG4Geometry);

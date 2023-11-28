// system include files
#include <map>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDSolidShapes.h"
#include "DetectorDescription/DDCMS/interface/DDCompactView.h"
#include "DetectorDescription/DDCMS/interface/DDDetector.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DD4hep/Detector.h"
#include "DD4hep/DD4hepRootPersistency.h"

#include "TGeoManager.h"
#include "TFile.h"
#include "TSystem.h"

class PrintGeomSolids : public edm::one::EDAnalyzer<> {
public:
  explicit PrintGeomSolids(const edm::ParameterSet&);
  ~PrintGeomSolids() override {}
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;

private:
  edm::ESGetToken<DDCompactView, IdealGeometryRecord> cpvTokenDDD_;
  edm::ESGetToken<cms::DDCompactView, IdealGeometryRecord> cpvTokenDD4hep_;
  bool fromDD4hep_;
};

PrintGeomSolids::PrintGeomSolids(const edm::ParameterSet& ps) {
  fromDD4hep_ = ps.getParameter<bool>("fromDD4hep");
  if (fromDD4hep_)
    cpvTokenDD4hep_ = esConsumes<cms::DDCompactView, IdealGeometryRecord>(edm::ESInputTag());
  else
    cpvTokenDDD_ = esConsumes<DDCompactView, IdealGeometryRecord>(edm::ESInputTag());

  edm::LogVerbatim("PrintGeom") << "PrintGeomSolids created with dd4hep: " << fromDD4hep_;
}

void PrintGeomSolids::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<bool>("fromDD4hep", false);
  descriptions.add("printGeomSolids", desc);
}

void PrintGeomSolids::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  int solids(0);
  std::map<std::string, int> shapes;
  if (fromDD4hep_) {
    const cms::DDCompactView* cpv = &iSetup.getData(cpvTokenDD4hep_);
    const cms::DDDetector* det = cpv->detector();
    TGeoManager const& geom = det->description()->manager();
    TGeoIterator next(geom.GetTopVolume());
    TGeoNode* node;
    TString path;
    std::vector<std::string> names;
    while ((node = next())) {
      next.GetPath(path);
      std::string name = static_cast<std::string>(node->GetVolume()->GetName());
      if (std::find(names.begin(), names.end(), name) == names.end()) {
        edm::LogVerbatim("PrintGeom") << name << "   "
                                      << static_cast<std::string>(node->GetVolume()->GetShape()->GetTitle());
        names.emplace_back(name);
        std::string shape = node->GetVolume()->GetShape()->GetTitle();
        if (shapes.find(shape) == shapes.end())
          shapes.emplace(std::make_pair(shape, 1));
        else
          ++shapes[shape];
        ++solids;
      }
    }

  } else {
    const DDCompactView* cpv = &iSetup.getData(cpvTokenDDD_);
    const auto& gra = cpv->graph();
    for (DDCompactView::Graph::const_adj_iterator git = gra.begin(); git != gra.end(); ++git) {
      const DDLogicalPart& ddLP = gra.nodeData(git);
      const DDSolid& solid = ddLP.solid();
      edm::LogVerbatim("PrintGeom") << solid.name() << "   " << DDSolidShapesName::name(solid.shape());
      std::string shape = DDSolidShapesName::name(solid.shape());
      if (shapes.find(shape) == shapes.end())
        shapes.emplace(std::make_pair(shape, 1));
      else
        ++shapes[shape];
      ++solids;
    }
  }
  edm::LogVerbatim("PrintGeom") << "\n\nPrintGeomSolids finds " << solids << " solids\n\n";
  for (std::map<std::string, int>::iterator itr = shapes.begin(); itr != shapes.end(); ++itr) {
    edm::LogVerbatim("PrintGeom") << "Shape:" << itr->first << " # " << itr->second;
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(PrintGeomSolids);

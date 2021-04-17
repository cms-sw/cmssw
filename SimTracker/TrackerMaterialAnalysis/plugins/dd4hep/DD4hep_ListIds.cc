#include <string>
#include <vector>
#include <iostream>
#include <algorithm>

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DetectorDescription/DDCMS/interface/DDFilteredView.h"
#include "DetectorDescription/DDCMS/interface/DDCompactView.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

class DD4hep_ListIds : public edm::one::EDAnalyzer<> {
public:
  DD4hep_ListIds(const edm::ParameterSet &);
  ~DD4hep_ListIds() override;

private:
  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void beginJob() override {}
  void endJob() override;

  // List of material names used to select specific detectors.
  // Names are matched literally, w/o any usage of regexp.
  // Names should also be specified with the correct namespace,
  // otherwise the matching will fail.
  bool printMaterial_;
  std::vector<std::string> materials_;
};

DD4hep_ListIds::DD4hep_ListIds(const edm::ParameterSet &pset)
    : printMaterial_(pset.getUntrackedParameter<bool>("printMaterial")),
      materials_(pset.getUntrackedParameter<std::vector<std::string> >("materials")) {}

DD4hep_ListIds::~DD4hep_ListIds() {}

void DD4hep_ListIds::analyze(const edm::Event &evt, const edm::EventSetup &setup) {
  edm::ESTransientHandle<cms::DDCompactView> cpv;
  setup.get<IdealGeometryRecord>().get("CMS", cpv);

  std::string attribute = "TkDDDStructure";
  cms::DDFilter filter(attribute, "");
  cms::DDFilteredView fv(*cpv, filter);
  fv.firstChild();
  std::set<std::string_view> tkdss;

  for (const auto &t : fv.specpars()) {
    tkdss.insert(t.second->strValue(attribute));
  }

  for (const auto &i : tkdss) {
    edm::LogVerbatim("ListIds") << "\nFiltering " << i;
    cms::DDFilter filter1(attribute, {i.data(), i.size()});
    cms::DDFilteredView fv1(*cpv, filter1);
    fv1.firstChild();

    std::vector<const cms::Node *> nodes;
    while (fv1.firstChild()) {
      nodes = fv1.geoHistory();
      if (std::find(materials_.begin(),
                    materials_.end(),
                    nodes[nodes.size() - 1]->GetVolume()->GetMaterial()->GetName()) != materials_.end()) {
        for (const auto &n : nodes) {
          edm::LogVerbatim("ListIds") << n->GetVolume()->GetName() << "[" << n->GetNumber() << "]/";
        }
        edm::LogVerbatim("ListIds") << "Material:|" << nodes[nodes.size() - 1]->GetVolume()->GetMaterial()->GetName()
                                    << "|\n";
      }
    }
  }
}

void DD4hep_ListIds::endJob() {}

//-------------------------------------------------------------------------
// define as a plugin
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(DD4hep_ListIds);

#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <iomanip>

#include "boost/format.hpp"

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/types.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DetectorDescription/DDCMS/interface/DDFilteredView.h"
#include "DetectorDescription/DDCMS/interface/DDCompactView.h"
#include "DetectorDescription/DDCMS/interface/DDSpecParRegistry.h"
#include "Geometry/Records/interface/DDSpecParRegistryRcd.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

class DD4hep_ListGroups : public edm::one::EDAnalyzer<> {
public:
  DD4hep_ListGroups(const edm::ParameterSet& iConfig);
  ~DD4hep_ListGroups() override;

private:
  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void beginJob() override {}
  void endJob() override;
  const edm::ESInputTag m_tag;

  bool m_saveSummaryPlot;
  std::vector<TH2F *> m_plots;
  std::set<std::string_view> m_group_names;
  std::vector<unsigned int> m_color;
  std::vector<int> m_gradient;

};

DD4hep_ListGroups::DD4hep_ListGroups(const edm::ParameterSet &iConfig) :
  m_tag(iConfig.getParameter<edm::ESInputTag>("DDDetector")) {
  m_saveSummaryPlot = iConfig.getUntrackedParameter<bool>("SaveSummaryPlot");
}

DD4hep_ListGroups::~DD4hep_ListGroups() {

}

void DD4hep_ListGroups::analyze(const edm::Event &evt, const edm::EventSetup &setup) {

  edm::ESTransientHandle<cms::DDCompactView> cpv;
  setup.get<IdealGeometryRecord>().get(m_tag,cpv);
  cms::DDFilteredView fv((*cpv).detector(),(*cpv).detector()->worldVolume());
  cms::DDSpecParRefs refs;
  const std::string_view tmg{"TrackingMaterialGroup"};
  const cms::DDSpecParRegistry& mypar = (*cpv).specpars();
  mypar.filter(refs, tmg);

  fv.mergedSpecifics(refs);

  for (const auto& t : refs) {
    m_group_names.insert(t->strValue(tmg.data()));
  }

}

void DD4hep_ListGroups::endJob() {}

//-------------------------------------------------------------------------
// define as a plugin
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(DD4hep_ListGroups);

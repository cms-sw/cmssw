#ifndef DD4hep_TrackingMaterialAnalyser_h
#define DD4hep_TrackingMaterialAnalyser_h
#include <string>
#include <vector>

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "SimDataFormats/ValidationFormats/interface/MaterialAccountingTrack.h"
#include "DD4hep_MaterialAccountingGroup.h"
#include "DD4hep_TrackingMaterialPlotter.h"

class DD4hep_TrackingMaterialAnalyser : public edm::one::EDAnalyzer<> {
public:
  explicit DD4hep_TrackingMaterialAnalyser(const edm::ParameterSet &);
  ~DD4hep_TrackingMaterialAnalyser() override;

private:
  enum SplitMode { NEAREST_LAYER, INNER_LAYER, OUTER_LAYER, UNDEFINED };

  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void beginJob() override {}
  void endJob() override;

  void split(MaterialAccountingTrack &track);
  int findLayer(const MaterialAccountingDetector &detector);

  void saveParameters(const char *name);
  void saveXml(const char *name);
  void saveLayerPlots();

  edm::EDGetTokenT<std::vector<MaterialAccountingTrack> > m_materialToken;
  edm::ESInputTag m_tag;
  SplitMode m_splitMode;
  bool m_skipAfterLastDetector;
  bool m_skipBeforeFirstDetector;
  bool m_saveSummaryPlot;
  bool m_saveDetailedPlots;
  bool m_saveParameters;
  bool m_saveXml;
  bool m_isHGCal;
  bool m_isHFNose;
  std::vector<DD4hep_MaterialAccountingGroup *> m_groups;
  std::vector<std::string> m_groupNames;
  std::unique_ptr<DD4hep_TrackingMaterialPlotter> m_plotter;
};

#endif  // DD4hep_TrackingMaterialAnalyser_h

#ifndef TrackingMaterialAnalyser_h
#define TrackingMaterialAnalyser_h
#include <string>
#include <vector>

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "SimDataFormats/ValidationFormats/interface/MaterialAccountingTrack.h"
#include "MaterialAccountingGroup.h"
#include "TrackingMaterialPlotter.h"

class TrackingMaterialAnalyser : public edm::EDAnalyzer
{
public:
  TrackingMaterialAnalyser(const edm::ParameterSet &);
  virtual ~TrackingMaterialAnalyser();
  
private:
  enum SplitMode {
    NEAREST_LAYER,
    INNER_LAYER,
    OUTER_LAYER,
    UNDEFINED
  };
  
  void analyze(const edm::Event &, const edm::EventSetup &);
  void beginJob(const edm::EventSetup &);
  void endJob();

  void split( MaterialAccountingTrack & track );
  int  findLayer( const MaterialAccountingDetector & detector );

  void save();
  void saveParameters(const char* name);
  void saveXml(const char* name);
  void saveLayerPlots(const char* name);
  
  edm::InputTag                             m_material;
  SplitMode                                 m_splitMode;
  bool                                      m_skipAfterLastDetector;
  bool                                      m_skipBeforeFirstDetector;
  bool                                      m_saveSummaryPlot;
  bool                                      m_saveDetailedPlots;
  bool                                      m_saveParameters;
  bool                                      m_saveXml;
  std::vector<MaterialAccountingGroup *>    m_groups;
  std::vector<std::string>                  m_groupNames;
  TrackingMaterialPlotter *                 m_plotter;

  edm::ESWatcher<IdealGeometryRecord>       m_geometryWatcher;
};

#endif // TrackingMaterialAnalyser_h

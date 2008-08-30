#ifndef TrackingMaterialAnalyser_h
#define TrackingMaterialAnalyser_h
#include <string>
#include <vector>

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "SimDataFormats/ValidationFormats/interface/MaterialAccountingTrack.h"
#include "MaterialAccountingLayer.h"
#include "TrackingMaterialPlotter.h"

class TrackingMaterialAnalyser : public edm::EDAnalyzer
{
public:
  TrackingMaterialAnalyser(const edm::ParameterSet &);
  virtual ~TrackingMaterialAnalyser();
  
private:
  void analyze(const edm::Event &, const edm::EventSetup &);
  void beginJob(const edm::EventSetup &);
  void endJob();

  void parseBarrelLayers( const std::vector<BarrelDetLayer*> & layers );
  void parseForwardLayers( const std::vector<ForwardDetLayer*> & neg_layers, const std::vector<ForwardDetLayer*> & pos_layers );
  
  void split( MaterialAccountingTrack & track );
  int  findLayer( const MaterialAccountingDetector & detector );

  void saveParameters(const char* name);
  void saveLayerPlots();
  
  edm::InputTag                             m_material;
  bool                                      m_skipAfterLastDetector;
  bool                                      m_skipBeforeFirstDetector;
  bool                                      m_symmetricForwardLayers;
  bool                                      m_saveSummaryPlot;
  bool                                      m_saveDetailedPlots;
  bool                                      m_saveParameters;
  std::vector<MaterialAccountingLayer *>    m_layers;
  TrackingMaterialPlotter *                 m_plotter;
};




#endif // TrackingMaterialAnalyser_h

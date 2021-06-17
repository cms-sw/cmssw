// Package:    Phase2OTValidateRecHitBase
// Class:      Phase2OTValidateRecHitBase
//
/**\class Phase2OTValidateRecHitBase Phase2OTValidateRecHitBase.cc 
 Description:  Standalone  Plugin for Phase2 RecHit validation
*/
//
// Author: Suvankar Roy Chowdhury
// Date: March 2021
//
// system include files
// STL includes
#include <memory>
#include <map>
#include <vector>
#include <algorithm>

// system include files
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"

//DQM
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DataFormats/TrackerRecHit2D/interface/Phase2TrackerRecHit1D.h"

class Phase2OTValidateRecHitBase : public DQMEDAnalyzer {
public:
  explicit Phase2OTValidateRecHitBase(const edm::ParameterSet&);
  ~Phase2OTValidateRecHitBase() override;
  void bookHistograms(DQMStore::IBooker& ibooker, edm::Run const& iRun, edm::EventSetup const& iSetup) override;
  void dqmBeginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) override;
  void fillOTRecHitHistos(const PSimHit* simhitClosest,
                          const Phase2TrackerRecHit1D* rechit,
                          const std::map<unsigned int, SimTrack>& selectedSimTrackMap,
                          std::map<std::string, unsigned int>& nrechitLayerMapP_primary,
                          std::map<std::string, unsigned int>& nrechitLayerMapS_primary);

  static void fillPSetDescription(edm::ParameterSetDescription& desc);
  void bookLayerHistos(DQMStore::IBooker& ibooker, unsigned int det_id, std::string& subdir);

protected:
  edm::ParameterSet config_;
  std::string geomType_;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topoToken_;
  const TrackerGeometry* tkGeom_ = nullptr;
  const TrackerTopology* tTopo_ = nullptr;

  struct RecHitME {
    // use TH1D instead of TH1F to avoid stauration at 2^31
    // above this increments with +1 don't work for float, need double
    MonitorElement* deltaX_P = nullptr;
    MonitorElement* deltaX_S = nullptr;
    MonitorElement* deltaY_P = nullptr;
    MonitorElement* deltaY_S = nullptr;
    MonitorElement* pullX_P = nullptr;
    MonitorElement* pullX_S = nullptr;
    MonitorElement* pullY_P = nullptr;
    MonitorElement* pullY_S = nullptr;
    MonitorElement* deltaX_eta_P = nullptr;
    MonitorElement* deltaX_eta_S = nullptr;
    MonitorElement* deltaY_eta_P = nullptr;
    MonitorElement* deltaY_eta_S = nullptr;
    MonitorElement* deltaX_phi_P = nullptr;
    MonitorElement* deltaX_phi_S = nullptr;
    MonitorElement* deltaY_phi_P = nullptr;
    MonitorElement* deltaY_phi_S = nullptr;
    MonitorElement* pullX_eta_P = nullptr;
    MonitorElement* pullX_eta_S = nullptr;
    MonitorElement* pullY_eta_P = nullptr;
    MonitorElement* pullY_eta_S = nullptr;
    //For rechits matched to simhits from highPT tracks
    MonitorElement* pullX_primary_P;
    MonitorElement* pullX_primary_S;
    MonitorElement* pullY_primary_P;
    MonitorElement* pullY_primary_S;
    MonitorElement* deltaX_primary_P;
    MonitorElement* deltaX_primary_S;
    MonitorElement* deltaY_primary_P;
    MonitorElement* deltaY_primary_S;
    MonitorElement* numberRecHitsprimary_P;
    MonitorElement* numberRecHitsprimary_S;
  };
  std::map<std::string, RecHitME> layerMEs_;
};

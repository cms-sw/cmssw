#ifndef Validation_SiTrackerPhase2_Phase2ITValidateRecHitBase_h
#define Validation_SiTrackerPhase2_Phase2ITValidateRecHitBase_h

/**\class Phase2ITValidateRecHitBase  
 Description:  Base Class for Phase2 Validation
*/
//
// Author: Marco Musich
// Date: May 2021
//

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

class Phase2ITValidateRecHitBase : public DQMEDAnalyzer {
public:
  explicit Phase2ITValidateRecHitBase(const edm::ParameterSet&);
  ~Phase2ITValidateRecHitBase() override;
  void bookHistograms(DQMStore::IBooker& ibooker, edm::Run const& iRun, edm::EventSetup const& iSetup) override;
  void dqmBeginRun(const edm::Run&, const edm::EventSetup&) override;
  static void fillPSetDescription(edm::ParameterSetDescription& desc);

protected:
  void bookLayerHistos(DQMStore::IBooker& ibooker, unsigned int det_id, std::string& subdir);
  void fillRechitHistos(const PSimHit* simhitClosest,
                        const SiPixelRecHit* rechit,
                        const std::map<unsigned int, SimTrack>& selectedSimTrackMap,
                        std::map<std::string, unsigned int>& nrechitLayerMap_primary);

  edm::ParameterSet config_;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topoToken_;
  const TrackerGeometry* tkGeom_ = nullptr;
  const TrackerTopology* tTopo_ = nullptr;

  struct RecHitME {
    MonitorElement* deltaX = nullptr;
    MonitorElement* deltaY = nullptr;
    MonitorElement* pullX = nullptr;
    MonitorElement* pullY = nullptr;
    MonitorElement* deltaX_eta = nullptr;
    MonitorElement* deltaX_phi = nullptr;
    MonitorElement* deltaY_eta = nullptr;
    MonitorElement* deltaY_phi = nullptr;
    MonitorElement* deltaX_clsizex = nullptr;
    MonitorElement* deltaX_clsizey = nullptr;
    MonitorElement* deltaY_clsizex = nullptr;
    MonitorElement* deltaY_clsizey = nullptr;
    MonitorElement* deltaYvsdeltaX = nullptr;
    MonitorElement* pullX_eta = nullptr;
    MonitorElement* pullY_eta = nullptr;
    //For rechits matched to primary simhits
    MonitorElement* numberRecHitsprimary = nullptr;
    MonitorElement* pullX_primary;
    MonitorElement* pullY_primary;
    MonitorElement* deltaX_primary;
    MonitorElement* deltaY_primary;
  };
  std::map<std::string, RecHitME> layerMEs_;
};

#endif

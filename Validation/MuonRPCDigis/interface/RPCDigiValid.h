#ifndef RPCDigiValid_h
#define RPCDigiValid_h

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <string>

#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

class RPCDigiValid : public DQMEDAnalyzer {
public:
  RPCDigiValid(const edm::ParameterSet &ps);
  ~RPCDigiValid() override = default;

protected:
  void analyze(const edm::Event &e, const edm::EventSetup &c) override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;

private:
  // RZ and XY views
  MonitorElement *hRZ_;

  MonitorElement *hXY_Barrel_;
  std::map<int, MonitorElement *> hXY_Endcap_;  // X-Y plots for Endcap, by station
  std::map<int, MonitorElement *> hZPhi_;       // R-phi plots for Barrel, by layers

  // Strip profile
  MonitorElement *hStripProf_;
  MonitorElement *hStripProf_RB12_, *hStripProf_RB34_;
  MonitorElement *hStripProf_Endcap_, *hStripProf_IRPC_;

  // Bunch crossing distributions
  MonitorElement *hBxDist_;
  MonitorElement *hBxDisc_4Plus_;
  MonitorElement *hBxDisc_4Min_;

  // Timing information
  bool isDigiTimeAvailable_;
  MonitorElement *hDigiTimeAll_, *hDigiTime_, *hDigiTimeIRPC_, *hDigiTimeNoIRPC_;

  // Multiplicity plots
  MonitorElement *hNSimHitPerRoll_, *hNDigiPerRoll_;

  // Residual plots
  MonitorElement *hRes_;
  std::map<int, MonitorElement *> hResBarrelLayers_;
  std::map<int, MonitorElement *> hResBarrelWheels_;
  std::map<int, MonitorElement *> hResEndcapDisks_;
  std::map<int, MonitorElement *> hResEndcapRings_;

  // Tokens for accessing run data. Used for passing to edm::Event. - stanislav
  edm::EDGetTokenT<edm::PSimHitContainer> simHitToken_;
  edm::EDGetTokenT<RPCDigiCollection> rpcDigiToken_;

  edm::ESGetToken<RPCGeometry, MuonGeometryRecord> rpcGeomToken_;
};

#endif

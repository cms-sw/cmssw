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
  MonitorElement *xyview;
  MonitorElement *rzview;
  MonitorElement *BxDist;
  MonitorElement *StripProf;

  MonitorElement *BxDisc_4Plus;
  MonitorElement *BxDisc_4Min;
  MonitorElement *xyvDplu4;
  MonitorElement *xyvDmin4;

  // Timing information
  MonitorElement *hDigiTimeAll, *hDigiTime, *hDigiTimeIRPC, *hDigiTimeNoIRPC;

  // Tokens for accessing run data. Used for passing to edm::Event. - stanislav
  edm::EDGetTokenT<edm::PSimHitContainer> simHitToken_;
  edm::EDGetTokenT<RPCDigiCollection> rpcDigiToken_;

  edm::ESGetToken<RPCGeometry, MuonGeometryRecord> rpcGeomToken_;
};

#endif

#include "Validation/MuonRPCDigis/interface/RPCDigiValid.h"

#include "FWCore/Utilities/interface/InputTag.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "Geometry/CommonTopologies/interface/RectangularStripTopology.h"
#include "Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h"

#include <cmath>

using namespace std;
using namespace edm;

RPCDigiValid::RPCDigiValid(const ParameterSet &ps) {
  //  Init the tokens for run data retrieval - stanislav
  //  ps.getUntackedParameter<InputTag> retrieves a InputTag from the
  //  configuration. The second param is default value module, instance and
  //  process labels may be passed in a single string if separated by colon ':'
  //  (@see the edm::InputTag constructor documentation)
  simHitToken_ = consumes<PSimHitContainer>(
      ps.getUntrackedParameter<edm::InputTag>("simHitTag", edm::InputTag("g4SimHits:MuonRPCHits")));
  rpcDigiToken_ = consumes<RPCDigiCollection>(
      ps.getUntrackedParameter<edm::InputTag>("rpcDigiTag", edm::InputTag("simMuonRPCDigis")));

  rpcGeomToken_ = esConsumes();
}

void RPCDigiValid::analyze(const Event &event, const EventSetup &eventSetup) {
  // Get the RPC Geometry
  auto rpcGeom = eventSetup.getHandle(rpcGeomToken_);

  edm::Handle<PSimHitContainer> simHitHandle;
  edm::Handle<RPCDigiCollection> rpcDigisHandle;
  event.getByToken(simHitToken_, simHitHandle);
  event.getByToken(rpcDigiToken_, rpcDigisHandle);

  // loop over Simhit
  for (auto simIt = simHitHandle->begin(); simIt != simHitHandle->end(); ++simIt) {
    const RPCDetId Rsid = simIt->detUnitId();
    const RPCRoll *roll = dynamic_cast<const RPCRoll *>(rpcGeom->roll(Rsid));
    if (!roll)
      continue;

    const GlobalPoint p = roll->toGlobal(simIt->localPosition());
    xyview->Fill(p.x(), p.y());

    if (Rsid.region() == (+1)) {
      if (Rsid.station() == 4) {
        xyvDplu4->Fill(p.x(), p.y());
      }
    } else if (Rsid.region() == (-1)) {
      if (Rsid.station() == 4) {
        xyvDmin4->Fill(p.x(), p.y());
      }
    }
    rzview->Fill(p.z(), p.perp());
  }
  // loop over Digis
  for (auto detUnitIt = rpcDigisHandle->begin(); detUnitIt != rpcDigisHandle->end(); ++detUnitIt) {
    const RPCDetId Rsid = (*detUnitIt).first;
    const RPCRoll *roll = dynamic_cast<const RPCRoll *>(rpcGeom->roll(Rsid));
    if (!roll)
      continue;

    const RPCDigiCollection::Range &range = (*detUnitIt).second;

    for (auto digiIt = range.first; digiIt != range.second; ++digiIt) {
      StripProf->Fill(digiIt->strip());
      BxDist->Fill(digiIt->bx());
      // bx for 4 endcaps
      if (Rsid.region() == (+1)) {
        if (Rsid.station() == 4)
          BxDisc_4Plus->Fill(digiIt->bx());
      } else if (Rsid.region() == (-1)) {
        if (Rsid.station() == 4)
          BxDisc_4Min->Fill(digiIt->bx());
      }

      // Fill timing information
      const double digiTime = digiIt->hasTime() ? digiIt->time() : digiIt->bx() * 25;
      hDigiTimeAll->Fill(digiTime);
      if (digiIt->hasTime()) {
        hDigiTime->Fill(digiTime);
        if (roll->isIRPC())
          hDigiTimeIRPC->Fill(digiTime);
        else
          hDigiTimeNoIRPC->Fill(digiTime);
      }
    }
  }
}

void RPCDigiValid::bookHistograms(DQMStore::IBooker &booker, edm::Run const &run, edm::EventSetup const &eSetup) {
  booker.setCurrentFolder("RPCDigisV/RPCDigis");

  xyview = booker.book2D("X_Vs_Y_View", "X_Vs_Y_View", 155, -775., 775., 155, -775., 775.);

  xyvDplu4 = booker.book2D("Dplu4_XvsY", "Dplu4_XvsY", 155, -775., 775., 155, -775., 775.);
  xyvDmin4 = booker.book2D("Dmin4_XvsY", "Dmin4_XvsY", 155, -775., 775., 155, -775., 775.);

  rzview = booker.book2D("R_Vs_Z_View", "R_Vs_Z_View", 216, -1080., 1080., 52, 260., 780.);

  BxDist = booker.book1D("Bunch_Crossing", "Bunch_Crossing", 20, -10., 10.);
  StripProf = booker.book1D("Strip_Profile", "Strip_Profile", 100, 0, 100);

  BxDisc_4Plus = booker.book1D("BxDisc_4Plus", "BxDisc_4Plus", 20, -10., 10.);
  BxDisc_4Min = booker.book1D("BxDisc_4Min", "BxDisc_4Min", 20, -10., 10.);

  // Timing informations
  hDigiTimeAll =
      booker.book1D("DigiTimeAll", "Digi time including present electronics;Digi time (ns)", 100, -12.5, 12.5);
  hDigiTime = booker.book1D("DigiTime", "Digi time only with timing information;Digi time (ns)", 100, -12.5, 12.5);
  hDigiTimeIRPC = booker.book1D("DigiTimeIRPC", "IRPC Digi time;Digi time (ns)", 100, -12.5, 12.5);
  hDigiTimeNoIRPC = booker.book1D("DigiTimeNoIRPC", "non-IRPC Digi time;Digi time (ns)", 100, -12.5, 12.5);
}

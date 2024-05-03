#include "Validation/MuonRPCDigis/interface/RPCDigiValid.h"

#include "FWCore/Utilities/interface/InputTag.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "Geometry/CommonTopologies/interface/RectangularStripTopology.h"
#include "Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h"

#include <cmath>
#include <fmt/format.h>

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
    const RPCDetId rsid = simIt->detUnitId();
    const RPCRoll *roll = dynamic_cast<const RPCRoll *>(rpcGeom->roll(rsid));
    if (!roll)
      continue;
    const int region = rsid.region();
    const GlobalPoint gp = roll->toGlobal(simIt->localPosition());

    hRZ_->Fill(gp.z(), gp.perp());

    if (region == 0) {
      // Barrel
      hXY_Barrel_->Fill(gp.x(), gp.y());
    } else {
      // Endcap
      const int disk = region * rsid.station();
      auto match = hXY_Endcap_.find(disk);
      if (match != hXY_Endcap_.end())
        match->second->Fill(gp.x(), gp.y());
    }
  }
  // loop over Digis
  for (auto detUnitIt = rpcDigisHandle->begin(); detUnitIt != rpcDigisHandle->end(); ++detUnitIt) {
    const RPCDetId rsid = (*detUnitIt).first;
    const RPCRoll *roll = dynamic_cast<const RPCRoll *>(rpcGeom->roll(rsid));
    if (!roll)
      continue;
    const int region = rsid.region();

    const RPCDigiCollection::Range &range = (*detUnitIt).second;

    for (auto digiIt = range.first; digiIt != range.second; ++digiIt) {
      // Strip profile
      const int strip = digiIt->strip();
      hStripProf->Fill(strip);

      if (region == 0) {
        // Barrel
        const int station = rsid.station();
        if (station == 1 or station == 2)
          hStripProf_RB12_->Fill(strip);
        else if (station == 3 or station == 4)
          hStripProf_RB34_->Fill(strip);
      } else {
        const int ring = rsid.ring();
        if (ring == 1)
          hStripProf_IRPC_->Fill(strip);
        else
          hStripProf_Endcap_->Fill(strip);
      }

      BxDist->Fill(digiIt->bx());
      // bx for 4 endcaps
      if (rsid.region() == (+1)) {
        if (rsid.station() == 4)
          BxDisc_4Plus->Fill(digiIt->bx());
      } else if (rsid.region() == (-1)) {
        if (rsid.station() == 4)
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

  // RZ plot
  hRZ_ = booker.book2D("RZ", "RZ", 220, -1100., 1100., 60, 0., 780.);

  // XY plots
  const int nbinsXY = 155;
  const double xmaxXY = 775;
  hXY_Barrel_ = booker.book2D("XY_Barrel", "XY_Barrel", nbinsXY, -xmaxXY, xmaxXY, nbinsXY, -xmaxXY, xmaxXY);
  for (int disk = 1; disk <= 4; ++disk) {
    const std::string meNameP = fmt::format("XY_EndcapP{:1d}", disk);
    const std::string meNameN = fmt::format("XY_EndcapN{:1d}", disk);
    hXY_Endcap_[disk] = booker.book2D(meNameP, meNameP, nbinsXY, -xmaxXY, xmaxXY, nbinsXY, -xmaxXY, xmaxXY);
    hXY_Endcap_[-disk] = booker.book2D(meNameN, meNameN, nbinsXY, -xmaxXY, xmaxXY, nbinsXY, -xmaxXY, xmaxXY);
  }

  // Strip profile
  hStripProf = booker.book1D("Strip_Profile", "Strip_Profile", 100, 0, 100);
  hStripProf_RB12_ = booker.book1D("Strip_Profile_RB12", "Strip Profile RB1 and RB2", 100, 0, 100);
  hStripProf_RB34_ = booker.book1D("Strip_Profile_RB12", "Strip Profile RB1 and RB2", 50, 0, 50);
  hStripProf_Endcap_ = booker.book1D("Strip_Profile_Endcap", "Strip Profile Endcap", 40, 0, 40);
  hStripProf_IRPC_ = booker.book1D("Strip_Profile_IRPC", "Strip Profile IRPC", 100, 0, 100);

  // Bunch crossing
  BxDist = booker.book1D("Bunch_Crossing", "Bunch_Crossing", 20, -10., 10.);
  BxDisc_4Plus = booker.book1D("BxDisc_4Plus", "BxDisc_4Plus", 20, -10., 10.);
  BxDisc_4Min = booker.book1D("BxDisc_4Min", "BxDisc_4Min", 20, -10., 10.);

  // Timing informations
  hDigiTimeAll =
      booker.book1D("DigiTimeAll", "Digi time including present electronics;Digi time (ns)", 100, -12.5, 12.5);
  hDigiTime = booker.book1D("DigiTime", "Digi time only with timing information;Digi time (ns)", 100, -12.5, 12.5);
  hDigiTimeIRPC = booker.book1D("DigiTimeIRPC", "IRPC Digi time;Digi time (ns)", 100, -12.5, 12.5);
  hDigiTimeNoIRPC = booker.book1D("DigiTimeNoIRPC", "non-IRPC Digi time;Digi time (ns)", 100, -12.5, 12.5);
}

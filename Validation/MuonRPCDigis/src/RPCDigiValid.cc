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

  isDigiTimeAvailable_ = ps.getUntrackedParameter<bool>("digiTime", false);

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
  std::map<const RPCRoll *, std::vector<double>> detToSimHitXsMap;
  for (auto simIt = simHitHandle->begin(); simIt != simHitHandle->end(); ++simIt) {
    const RPCDetId rsid = simIt->detUnitId();
    const RPCRoll *roll = dynamic_cast<const RPCRoll *>(rpcGeom->roll(rsid));
    if (!roll)
      continue;

    if (detToSimHitXsMap.find(roll) == detToSimHitXsMap.end())
      detToSimHitXsMap[roll] = std::vector<double>();
    detToSimHitXsMap[roll].push_back(simIt->localPosition().x());

    const int region = rsid.region();
    const GlobalPoint gp = roll->toGlobal(simIt->localPosition());

    hRZ_->Fill(gp.z(), gp.perp());

    if (region == 0) {
      // Barrel
      hXY_Barrel_->Fill(gp.x(), gp.y());

      const int station = rsid.station();
      const int layer = rsid.layer();
      const int stla = (station <= 2) ? (2 * (station - 1) + layer) : (station + 2);

      auto match = hZPhi_.find(stla);
      if (match != hZPhi_.end()) {
        const double phiInDeg = 180. * gp.barePhi() / TMath::Pi();
        match->second->Fill(gp.z(), phiInDeg);
      }
    } else {
      // Endcap
      const int disk = region * rsid.station();
      auto match = hXY_Endcap_.find(disk);
      if (match != hXY_Endcap_.end())
        match->second->Fill(gp.x(), gp.y());
    }
  }
  for (const auto &detToSimHitXs : detToSimHitXsMap) {
    hNSimHitPerRoll_->Fill(detToSimHitXs.second.size());
  }

  // loop over Digis
  std::map<const RPCRoll *, std::vector<double>> detToDigiXsMap;
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
      hStripProf_->Fill(strip);

      if (region == 0) {
        const int station = rsid.station();
        if (station == 1 or station == 2)
          hStripProf_RB12_->Fill(strip);
        else if (station == 3 or station == 4)
          hStripProf_RB34_->Fill(strip);
      } else {
        if (roll->isIRPC())
          hStripProf_IRPC_->Fill(strip);
        else
          hStripProf_Endcap_->Fill(strip);
      }

      // Bunch crossing
      const int bx = digiIt->bx();
      hBxDist_->Fill(bx);
      // bx for 4 endcaps
      if (rsid.station() == 4) {
        if (region == 1) {
          hBxDisc_4Plus_->Fill(bx);
        } else if (region == -1) {
          hBxDisc_4Min_->Fill(bx);
        }
      }

      // Fill timing information
      if (isDigiTimeAvailable_) {
        const double digiTime = digiIt->hasTime() ? digiIt->time() : digiIt->bx() * 25;
        hDigiTimeAll_->Fill(digiTime);
        if (digiIt->hasTime()) {
          hDigiTime_->Fill(digiTime);
          if (roll->isIRPC())
            hDigiTimeIRPC_->Fill(digiTime);
          else
            hDigiTimeNoIRPC_->Fill(digiTime);
        }
      }

      // Keep digi position
      const double digiX = roll->centreOfStrip(digiIt->strip()).x();
      if (detToDigiXsMap.find(roll) == detToDigiXsMap.end())
        detToDigiXsMap[roll] = std::vector<double>();
      detToDigiXsMap[roll].push_back(digiX);
    }
  }
  for (const auto &detToDigiXs : detToDigiXsMap) {
    const auto digiXs = detToDigiXs.second;
    const int nDigi = digiXs.size();
    hNDigiPerRoll_->Fill(nDigi);

    // Fill residual plots, only for nDigi==1 and nSimHit==1
    const auto roll = detToDigiXs.first;
    const auto detId = roll->id();
    if (nDigi != 1)
      continue;
    if (detToSimHitXsMap.find(roll) == detToSimHitXsMap.end())
      continue;

    const auto simHitXs = detToSimHitXsMap[roll];
    const int nSimHit = simHitXs.size();
    if (nSimHit != 1)
      continue;

    const double dx = digiXs[0] - simHitXs[0];
    hRes_->Fill(dx);
    if (roll->isBarrel()) {
      const int wheel = detId.ring();  // ring() is wheel number for Barrel
      const int station = detId.station();
      const int layer = detId.layer();
      const int stla = (station <= 2) ? (2 * (station - 1) + layer) : (station + 2);

      auto matchLayer = hResBarrelLayers_.find(stla);
      if (matchLayer != hResBarrelLayers_.end())
        matchLayer->second->Fill(dx);

      auto matchWheel = hResBarrelWheels_.find(wheel);
      if (matchWheel != hResBarrelWheels_.end())
        matchWheel->second->Fill(dx);
    } else {
      const int disk = detId.region() * detId.station();
      auto matchDisk = hResEndcapDisks_.find(disk);
      if (matchDisk != hResEndcapDisks_.end())
        matchDisk->second->Fill(dx);

      auto matchRing = hResEndcapRings_.find(detId.ring());
      if (matchRing != hResEndcapRings_.end())
        matchRing->second->Fill(dx);
    }
  }
}

void RPCDigiValid::bookHistograms(DQMStore::IBooker &booker, edm::Run const &run, edm::EventSetup const &eSetup) {
  booker.setCurrentFolder("RPCDigisV/RPCDigis");

  // Define binnings of 2D-histograms
  const double maxZ = 1100;
  const int nbinsZ = 220;  // bin width: 10cm
  const double maxXY = 800;
  const int nbinsXY = 160;  // bin width: 10cm
  const double minR = 100, maxR = 800;
  const int nbinsR = 70;    // bin width: 10cm
  const int nbinsPhi = 72;  // bin width: 5 degree
  const double maxBarrelZ = 700;
  const int nbinsBarrelZ = 140;  // bin width: 10cm

  // RZ plot
  hRZ_ = booker.book2D("RZ", "R-Z view;Z (cm);R (cm)", nbinsZ, -maxZ, maxZ, nbinsR, minR, maxR);
  hRZ_->setOption("colz");

  // XY plots
  hXY_Barrel_ = booker.book2D("XY_Barrel", "X-Y view of Barrel", nbinsXY, -maxXY, maxXY, nbinsXY, -maxXY, maxXY);
  hXY_Barrel_->setOption("colz");
  for (int disk = 1; disk <= 4; ++disk) {
    const std::string meNameP = fmt::format("XY_Endcap_p{:1d}", disk);
    const std::string meNameN = fmt::format("XY_Endcap_m{:1d}", disk);
    const std::string meTitleP = fmt::format("X-Y view of Endcap{:+1d};X (cm);Y (cm)", disk);
    const std::string meTitleN = fmt::format("X-Y view of Endcap{:+1d};X (cm);Y (cm)", -disk);
    hXY_Endcap_[disk] = booker.book2D(meNameP, meTitleP, nbinsXY, -maxXY, maxXY, nbinsXY, -maxXY, maxXY);
    hXY_Endcap_[-disk] = booker.book2D(meNameN, meTitleN, nbinsXY, -maxXY, maxXY, nbinsXY, -maxXY, maxXY);
    hXY_Endcap_[disk]->setOption("colz");
    hXY_Endcap_[-disk]->setOption("colz");
  }

  // Z-phi plots
  for (int layer = 1; layer <= 6; ++layer) {
    const std::string meName = fmt::format("ZPhi_Layer{:1d}", layer);
    const std::string meTitle = fmt::format("Z-#phi view of Layer{:1d};Z (cm);#phi (degree)", layer);
    hZPhi_[layer] = booker.book2D(meName, meTitle, nbinsBarrelZ, -maxBarrelZ, maxBarrelZ, nbinsPhi, -180, 180);
    hZPhi_[layer]->setOption("colz");
  }

  // Strip profile
  hStripProf_ = booker.book1D("Strip_Profile", "Strip_Profile;Strip Number", 100, 0, 100);
  hStripProf_RB12_ = booker.book1D("Strip_Profile_RB12", "Strip Profile RB1 and RB2;Strip Number", 92, 0, 92);
  hStripProf_RB34_ = booker.book1D("Strip_Profile_RB34", "Strip Profile RB3 and RB4;Strip Number", 62, 0, 62);
  hStripProf_Endcap_ = booker.book1D("Strip_Profile_Endcap", "Strip Profile Endcap;Strip Number", 40, 0, 40);
  hStripProf_IRPC_ = booker.book1D("Strip_Profile_IRPC", "Strip Profile IRPC;Strip Number", 100, 0, 100);

  // Bunch crossing
  hBxDist_ = booker.book1D("Bunch_Crossing", "Bunch Crossing;Bunch crossing", 20, -10., 10.);
  hBxDisc_4Plus_ = booker.book1D("BxDisc_4Plus", "BxDisc_4Plus", 20, -10., 10.);
  hBxDisc_4Min_ = booker.book1D("BxDisc_4Min", "BxDisc_4Min", 20, -10., 10.);

  // Timing informations
  if (isDigiTimeAvailable_) {
    hDigiTimeAll_ =
        booker.book1D("DigiTimeAll", "Digi time including present electronics;Digi time (ns)", 100, -12.5, 12.5);
    hDigiTime_ = booker.book1D("DigiTime", "Digi time only with timing information;Digi time (ns)", 100, -12.5, 12.5);
    hDigiTimeIRPC_ = booker.book1D("DigiTimeIRPC", "IRPC Digi time;Digi time (ns)", 100, -12.5, 12.5);
    hDigiTimeNoIRPC_ = booker.book1D("DigiTimeNoIRPC", "non-IRPC Digi time;Digi time (ns)", 100, -12.5, 12.5);
  }

  // SimHit and Digi multiplicity per roll
  hNSimHitPerRoll_ = booker.book1D("NSimHitPerRoll", "SimHit multiplicity per Roll;Multiplicity", 10, 0, 10);
  hNDigiPerRoll_ = booker.book1D("NDigiPerRoll", "Digi multiplicity per Roll;Multiplicity", 10, 0, 10);

  // Residual of SimHit-Digi x-position
  hRes_ = booker.book1D("Digi_SimHit_Difference", "Digi-SimHit difference;dx (cm)", 100, -8, 8);

  for (int layer = 1; layer <= 6; ++layer) {
    const std::string meName = fmt::format("Residual_Barrel_Layer{:1d}", layer);
    const std::string meTitle = fmt::format("Residual of Barrel Layer{:1d};dx (cm)", layer);
    hResBarrelLayers_[layer] = booker.book1D(meName, meTitle, 100, -8, 8);
  }

  hResBarrelWheels_[-2] = booker.book1D("Residual_Barrel_Wheel_m2", "Residual of Barrel Wheel-2;dx (cm)", 100, -8, 8);
  hResBarrelWheels_[-1] = booker.book1D("Residual_Barrel_Wheel_m1", "Residual of Barrel Wheel-1;dx (cm)", 100, -8, 8);
  hResBarrelWheels_[+0] = booker.book1D("Residual_Barrel_Wheel_00", "Residual of Barrel Wheel 0;dx (cm)", 100, -8, 8);
  hResBarrelWheels_[+1] = booker.book1D("Residual_Barrel_Wheel_p1", "Residual of Barrel Wheel+1;dx (cm)", 100, -8, 8);
  hResBarrelWheels_[+2] = booker.book1D("Residual_Barrel_Wheel_p2", "Residual of Barrel Wheel+2;dx (cm)", 100, -8, 8);

  for (int disk = 1; disk <= 4; ++disk) {
    const std::string meNameP = fmt::format("Residual_Endcap_Disk_p{:1d}", disk);
    const std::string meNameN = fmt::format("Residual_Endcap_Disk_m{:1d}", disk);
    const std::string meTitleP = fmt::format("Residual of Endcap Disk{:+1d};dx (cm)", disk);
    const std::string meTitleN = fmt::format("Residual of Endcap Disk{:+1d};dx (cm)", -disk);
    hResEndcapDisks_[+disk] = booker.book1D(meNameP, meTitleP, 100, -8, 8);
    hResEndcapDisks_[-disk] = booker.book1D(meNameN, meTitleN, 100, -8, 8);
  }

  hResEndcapRings_[1] = booker.book1D("Residual_Endcap_Ring1", "Residual of Endcap Ring1;dx (cm)", 100, -12, 12);
  hResEndcapRings_[2] = booker.book1D("Residual_Endcap_Ring2", "Residual of Endcap Ring2;dx (cm)", 100, -8, 8);
  hResEndcapRings_[3] = booker.book1D("Residual_Endcap_Ring3", "Residual of Endcap Ring3;dx (cm)", 100, -8, 8);
}

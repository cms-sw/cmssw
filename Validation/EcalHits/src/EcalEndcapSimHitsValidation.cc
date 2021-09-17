/*
 * \file EcalEndcapSimHitsValidation.cc
 *
 * \author C.Rovelli
 *
 */

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "Validation/EcalHits/interface/EcalEndcapSimHitsValidation.h"
#include <DataFormats/EcalDetId/interface/EEDetId.h>

using namespace cms;
using namespace edm;
using namespace std;

EcalEndcapSimHitsValidation::EcalEndcapSimHitsValidation(const edm::ParameterSet &ps)
    : g4InfoLabel(ps.getParameter<std::string>("moduleLabelG4")),
      EEHitsCollection(ps.getParameter<std::string>("EEHitsCollection")),
      ValidationCollection(ps.getParameter<std::string>("ValidationCollection")) {
  EEHitsToken =
      consumes<edm::PCaloHitContainer>(edm::InputTag(std::string(g4InfoLabel), std::string(EEHitsCollection)));
  ValidationCollectionToken =
      consumes<PEcalValidInfo>(edm::InputTag(std::string(g4InfoLabel), std::string(ValidationCollection)));
  // verbosity switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);

  myEntries = 0;
  for (int myStep = 0; myStep < 26; myStep++) {
    eRLength[myStep] = 0.0;
  }
}

void EcalEndcapSimHitsValidation::bookHistograms(DQMStore::IBooker &ib, edm::Run const &, edm::EventSetup const &c) {
  ib.setCurrentFolder("EcalHitsV/EcalSimHitsValidation");
  ib.setScope(MonitorElementData::Scope::RUN);

  std::string histo = "EE+ hits multiplicity";
  meEEzpHits_ = ib.book1D(histo, histo, 50, 0., 5000.);

  histo = "EE- hits multiplicity";
  meEEzmHits_ = ib.book1D(histo, histo, 50, 0., 5000.);

  histo = "EE+ crystals multiplicity";
  meEEzpCrystals_ = ib.book1D(histo, histo, 200, 0., 2000.);

  histo = "EE- crystals multiplicity";
  meEEzmCrystals_ = ib.book1D(histo, histo, 200, 0., 2000.);

  histo = "EE+ occupancy";
  meEEzpOccupancy_ = ib.book2D(histo, histo, 100, 0., 100., 100, 0., 100.);

  histo = "EE- occupancy";
  meEEzmOccupancy_ = ib.book2D(histo, histo, 100, 0., 100., 100, 0., 100.);

  histo = "EE longitudinal shower profile";
  meEELongitudinalShower_ = ib.bookProfile(histo, histo, 26, 0, 26, 100, 0, 20000);

  histo = "EE hits energy spectrum";
  meEEHitEnergy_ = ib.book1D(histo, histo, 4000, 0., 400.);

  histo = "EE hits log10energy spectrum";
  meEEhitLog10Energy_ = ib.book1D(histo, histo, 140, -10., 4.);

  histo = "EE hits log10energy spectrum vs normalized energy";
  meEEhitLog10EnergyNorm_ = ib.bookProfile(histo, histo, 140, -10., 4., 100, 0., 1.);

  histo = "EE hits log10energy spectrum vs normalized energy25";
  meEEhitLog10Energy25Norm_ = ib.bookProfile(histo, histo, 140, -10., 4., 100, 0., 1.);

  histo = "EE hits energy spectrum 2";
  meEEHitEnergy2_ = ib.book1D(histo, histo, 1000, 0., 0.001);

  histo = "EE crystal energy spectrum";
  meEEcrystalEnergy_ = ib.book1D(histo, histo, 5000, 0., 50.);

  histo = "EE crystal energy spectrum 2";
  meEEcrystalEnergy2_ = ib.book1D(histo, histo, 1000, 0., 0.001);

  histo = "EE E1";
  meEEe1_ = ib.book1D(histo, histo, 400, 0., 400.);

  histo = "EE E4";
  meEEe4_ = ib.book1D(histo, histo, 400, 0., 400.);

  histo = "EE E9";
  meEEe9_ = ib.book1D(histo, histo, 400, 0., 400.);

  histo = "EE E16";
  meEEe16_ = ib.book1D(histo, histo, 400, 0., 400.);

  histo = "EE E25";
  meEEe25_ = ib.book1D(histo, histo, 400, 0., 400.);

  histo = "EE E1oE4";
  meEEe1oe4_ = ib.book1D(histo, histo, 100, 0.4, 1.1);

  histo = "EE E1oE9";
  meEEe1oe9_ = ib.book1D(histo, histo, 100, 0.4, 1.1);

  histo = "EE E4oE9";
  meEEe4oe9_ = ib.book1D(histo, histo, 100, 0.4, 1.1);

  histo = "EE E9oE16";
  meEEe9oe16_ = ib.book1D(histo, histo, 100, 0.4, 1.1);

  histo = "EE E1oE25";
  meEEe1oe25_ = ib.book1D(histo, histo, 100, 0.4, 1.1);

  histo = "EE E9oE25";
  meEEe9oe25_ = ib.book1D(histo, histo, 100, 0.4, 1.1);

  histo = "EE E16oE25";
  meEEe16oe25_ = ib.book1D(histo, histo, 100, 0.4, 1.1);
}

void EcalEndcapSimHitsValidation::analyze(const edm::Event &e, const edm::EventSetup &c) {
  edm::LogInfo("EventInfo") << " Run = " << e.id().run() << " Event = " << e.id().event();

  edm::Handle<edm::PCaloHitContainer> EcalHitsEE;
  e.getByToken(EEHitsToken, EcalHitsEE);

  // Do nothing if no EndCap data available
  if (!EcalHitsEE.isValid())
    return;

  edm::Handle<PEcalValidInfo> MyPEcalValidInfo;
  e.getByToken(ValidationCollectionToken, MyPEcalValidInfo);

  std::vector<PCaloHit> theEECaloHits;
  theEECaloHits.insert(theEECaloHits.end(), EcalHitsEE->begin(), EcalHitsEE->end());

  myEntries++;

  std::map<unsigned int, std::vector<PCaloHit *>, std::less<unsigned int>> CaloHitMap;

  double EEetzp_ = 0.;
  double EEetzm_ = 0.;

  double ee1 = 0.0;
  double ee4 = 0.0;
  double ee9 = 0.0;
  double ee16 = 0.0;
  double ee25 = 0.0;
  std::vector<double> econtr(140, 0.);
  std::vector<double> econtr25(140, 0.);

  MapType eemap;
  MapType eemapzp;
  MapType eemapzm;
  uint32_t nEEzpHits = 0;
  uint32_t nEEzmHits = 0;

  for (std::vector<PCaloHit>::iterator isim = theEECaloHits.begin(); isim != theEECaloHits.end(); ++isim) {
    if (isim->time() > 500.) {
      continue;
    }

    CaloHitMap[isim->id()].push_back(&(*isim));

    EEDetId eeid(isim->id());

    LogDebug("HitInfo") << " CaloHit " << isim->getName() << "\n"
                        << " DetID = " << isim->id() << " EEDetId = " << eeid.ix() << " " << eeid.iy() << "\n"
                        << " Time = " << isim->time() << "\n"
                        << " Track Id = " << isim->geantTrackId() << "\n"
                        << " Energy = " << isim->energy();

    uint32_t crystid = eeid.rawId();

    if (eeid.zside() > 0) {
      nEEzpHits++;
      EEetzp_ += isim->energy();
      eemapzp[crystid] += isim->energy();
      meEEzpOccupancy_->Fill(eeid.ix(), eeid.iy());
    } else if (eeid.zside() < 0) {
      nEEzmHits++;
      EEetzm_ += isim->energy();
      eemapzm[crystid] += isim->energy();
      meEEzmOccupancy_->Fill(eeid.ix(), eeid.iy());
    }

    meEEHitEnergy_->Fill(isim->energy());
    if (isim->energy() > 0) {
      meEEhitLog10Energy_->Fill(log10(isim->energy()));
      int log10i = int((log10(isim->energy()) + 10.) * 10.);
      if (log10i >= 0 && log10i < 140)
        econtr[log10i] += isim->energy();
    }
    meEEHitEnergy2_->Fill(isim->energy());
    eemap[crystid] += isim->energy();
  }

  meEEzpCrystals_->Fill(eemapzp.size());
  meEEzmCrystals_->Fill(eemapzm.size());

  for (std::map<uint32_t, float, std::less<uint32_t>>::iterator it = eemap.begin(); it != eemap.end(); ++it)
    meEEcrystalEnergy_->Fill((*it).second);
  for (std::map<uint32_t, float, std::less<uint32_t>>::iterator it = eemap.begin(); it != eemap.end(); ++it)
    meEEcrystalEnergy2_->Fill((*it).second);

  meEEzpHits_->Fill(nEEzpHits);
  meEEzmHits_->Fill(nEEzmHits);

  int nEEHits = nEEzmHits + nEEzpHits;
  if (nEEHits > 0) {
    uint32_t eecenterid = getUnitWithMaxEnergy(eemap);
    EEDetId myEEid(eecenterid);
    int bx = myEEid.ix();
    int by = myEEid.iy();
    int bz = myEEid.zside();
    ee1 = energyInMatrixEE(1, 1, bx, by, bz, eemap);
    meEEe1_->Fill(ee1);
    ee9 = energyInMatrixEE(3, 3, bx, by, bz, eemap);
    meEEe9_->Fill(ee9);
    ee25 = energyInMatrixEE(5, 5, bx, by, bz, eemap);
    meEEe25_->Fill(ee25);

    std::vector<uint32_t> ids25;
    ids25 = getIdsAroundMax(5, 5, bx, by, bz, eemap);

    for (unsigned i = 0; i < 25; i++) {
      for (unsigned int j = 0; j < CaloHitMap[ids25[i]].size(); j++) {
        if (CaloHitMap[ids25[i]][j]->energy() > 0) {
          int log10i = int((log10(CaloHitMap[ids25[i]][j]->energy()) + 10.) * 10.);
          if (log10i >= 0 && log10i < 140)
            econtr25[log10i] += CaloHitMap[ids25[i]][j]->energy();
        }
      }
    }

    MapType neweemap;
    if (fillEEMatrix(3, 3, bx, by, bz, neweemap, eemap)) {
      ee4 = eCluster2x2(neweemap);
      meEEe4_->Fill(ee4);
    }
    if (fillEEMatrix(5, 5, bx, by, bz, neweemap, eemap)) {
      ee16 = eCluster4x4(ee9, neweemap);
      meEEe16_->Fill(ee16);
    }

    if (ee4 > 0.1)
      meEEe1oe4_->Fill(ee1 / ee4);
    if (ee9 > 0.1)
      meEEe1oe9_->Fill(ee1 / ee9);
    if (ee9 > 0.1)
      meEEe4oe9_->Fill(ee4 / ee9);
    if (ee16 > 0.1)
      meEEe9oe16_->Fill(ee9 / ee16);
    if (ee25 > 0.1)
      meEEe16oe25_->Fill(ee16 / ee25);
    if (ee25 > 0.1)
      meEEe1oe25_->Fill(ee1 / ee25);
    if (ee25 > 0.1)
      meEEe9oe25_->Fill(ee9 / ee25);

    if ((EEetzp_ + EEetzm_) != 0) {
      for (int i = 0; i < 140; i++) {
        meEEhitLog10EnergyNorm_->Fill(-10. + (float(i) + 0.5) / 10., econtr[i] / (EEetzp_ + EEetzm_));
      }
    }

    if (ee25 != 0) {
      for (int i = 0; i < 140; i++) {
        meEEhitLog10Energy25Norm_->Fill(-10. + (float(i) + 0.5) / 10., econtr25[i] / ee25);
      }
    }
  }

  if (MyPEcalValidInfo.isValid()) {
    if (MyPEcalValidInfo->ee1x1() > 0.) {
      std::vector<float> EX0 = MyPEcalValidInfo->eX0();
      meEELongitudinalShower_->Reset();
      for (int myStep = 0; myStep < 26; myStep++) {
        eRLength[myStep] += EX0[myStep];
        meEELongitudinalShower_->Fill(float(myStep), eRLength[myStep] / myEntries);
      }
    }
  }
}

float EcalEndcapSimHitsValidation::energyInMatrixEE(
    int nCellInX, int nCellInY, int centralX, int centralY, int centralZ, MapType &themap) {
  int ncristals = 0;
  float totalEnergy = 0.;

  int goBackInX = nCellInX / 2;
  int goBackInY = nCellInY / 2;
  int startX = centralX - goBackInX;
  int startY = centralY - goBackInY;

  for (int ix = startX; ix < startX + nCellInX; ix++) {
    for (int iy = startY; iy < startY + nCellInY; iy++) {
      uint32_t index;

      if (EEDetId::validDetId(ix, iy, centralZ)) {
        index = EEDetId(ix, iy, centralZ).rawId();
      } else {
        continue;
      }

      totalEnergy += themap[index];
      ncristals += 1;
    }
  }

  LogDebug("GeomInfo") << nCellInX << " x " << nCellInY << " EE matrix energy = " << totalEnergy << " for " << ncristals
                       << " crystals";
  return totalEnergy;
}

std::vector<uint32_t> EcalEndcapSimHitsValidation::getIdsAroundMax(
    int nCellInX, int nCellInY, int centralX, int centralY, int centralZ, MapType &themap) {
  int ncristals = 0;
  std::vector<uint32_t> ids(nCellInX * nCellInY);

  int goBackInX = nCellInX / 2;
  int goBackInY = nCellInY / 2;
  int startX = centralX - goBackInX;
  int startY = centralY - goBackInY;

  for (int ix = startX; ix < startX + nCellInX; ix++) {
    for (int iy = startY; iy < startY + nCellInY; iy++) {
      uint32_t index;

      if (EEDetId::validDetId(ix, iy, centralZ)) {
        index = EEDetId(ix, iy, centralZ).rawId();
      } else {
        continue;
      }

      ids[ncristals] = index;
      ncristals += 1;
    }
  }

  return ids;
}

bool EcalEndcapSimHitsValidation::fillEEMatrix(
    int nCellInX, int nCellInY, int CentralX, int CentralY, int CentralZ, MapType &fillmap, MapType &themap) {
  int goBackInX = nCellInX / 2;
  int goBackInY = nCellInY / 2;

  int startX = CentralX - goBackInX;
  int startY = CentralY - goBackInY;

  int i = 0;
  for (int ix = startX; ix < startX + nCellInX; ix++) {
    for (int iy = startY; iy < startY + nCellInY; iy++) {
      uint32_t index;

      if (EEDetId::validDetId(ix, iy, CentralZ)) {
        index = EEDetId(ix, iy, CentralZ).rawId();
      } else {
        continue;
      }
      fillmap[i++] = themap[index];
    }
  }
  uint32_t centerid = getUnitWithMaxEnergy(themap);

  if (fillmap[i / 2] == themap[centerid])
    return true;
  else
    return false;
}

float EcalEndcapSimHitsValidation::eCluster2x2(MapType &themap) {
  float E22 = 0.;
  float e012 = themap[0] + themap[1] + themap[2];
  float e036 = themap[0] + themap[3] + themap[6];
  float e678 = themap[6] + themap[7] + themap[8];
  float e258 = themap[2] + themap[5] + themap[8];

  if ((e012 > e678 || e012 == e678) && (e036 > e258 || e036 == e258))
    return E22 = themap[0] + themap[1] + themap[3] + themap[4];
  else if ((e012 > e678 || e012 == e678) && (e036 < e258 || e036 == e258))
    return E22 = themap[1] + themap[2] + themap[4] + themap[5];
  else if ((e012 < e678 || e012 == e678) && (e036 > e258 || e036 == e258))
    return E22 = themap[3] + themap[4] + themap[6] + themap[7];
  else if ((e012 < e678 || e012 == e678) && (e036 < e258 || e036 == e258))
    return E22 = themap[4] + themap[5] + themap[7] + themap[8];
  else {
    return E22;
  }
}

float EcalEndcapSimHitsValidation::eCluster4x4(float e33, MapType &themap) {
  float E44 = 0.;
  float e0_4 = themap[0] + themap[1] + themap[2] + themap[3] + themap[4];
  float e0_20 = themap[0] + themap[5] + themap[10] + themap[15] + themap[20];
  float e4_24 = themap[4] + themap[9] + themap[14] + themap[19] + themap[24];
  float e0_24 = themap[20] + themap[21] + themap[22] + themap[23] + themap[24];

  if ((e0_4 > e0_24 || e0_4 == e0_24) && (e0_20 > e4_24 || e0_20 == e4_24))
    return E44 = e33 + themap[0] + themap[1] + themap[2] + themap[3] + themap[5] + themap[10] + themap[15];
  else if ((e0_4 > e0_24 || e0_4 == e0_24) && (e0_20 < e4_24 || e0_20 == e4_24))
    return E44 = e33 + themap[1] + themap[2] + themap[3] + themap[4] + themap[9] + themap[14] + themap[19];
  else if ((e0_4 < e0_24 || e0_4 == e0_24) && (e0_20 > e4_24 || e0_20 == e4_24))
    return E44 = e33 + themap[5] + themap[10] + themap[15] + themap[20] + themap[21] + themap[22] + themap[23];
  else if ((e0_4 < e0_24 || e0_4 == e0_24) && (e0_20 < e4_24 || e0_20 == e4_24))
    return E44 = e33 + themap[21] + themap[22] + themap[23] + themap[24] + themap[9] + themap[14] + themap[19];
  else {
    return E44;
  }
}

uint32_t EcalEndcapSimHitsValidation::getUnitWithMaxEnergy(MapType &themap) {
  // look for max
  uint32_t unitWithMaxEnergy = 0;
  float maxEnergy = 0.;

  MapType::iterator iter;
  for (iter = themap.begin(); iter != themap.end(); iter++) {
    if (maxEnergy < (*iter).second) {
      maxEnergy = (*iter).second;
      unitWithMaxEnergy = (*iter).first;
    }
  }

  LogDebug("GeomInfo") << " max energy of " << maxEnergy << " GeV in Unit id " << unitWithMaxEnergy;
  return unitWithMaxEnergy;
}

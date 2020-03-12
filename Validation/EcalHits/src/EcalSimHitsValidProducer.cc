#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Validation/EcalHits/interface/EcalSimHitsValidProducer.h"

#include "DataFormats/Math/interface/Point3D.h"
#include "SimDataFormats/ValidationFormats/interface/PValidationFormats.h"
#include "SimG4CMS/Calo/interface/CaloG4HitCollection.h"
#include "SimG4Core/Notification/interface/BeginOfEvent.h"
#include "SimG4Core/Notification/interface/BeginOfTrack.h"
#include "SimG4Core/Notification/interface/EndOfEvent.h"

#include "G4PrimaryParticle.hh"
#include "G4PrimaryVertex.hh"
#include "G4SDManager.hh"
#include "G4Step.hh"

#include <iostream>

EcalSimHitsValidProducer::EcalSimHitsValidProducer(const edm::ParameterSet &iPSet)
    : ee1(0.0),
      ee4(0.0),
      ee9(0.0),
      ee16(0.0),
      ee25(0.0),
      eb1(0.0),
      eb4(0.0),
      eb9(0.0),
      eb16(0.0),
      eb25(0.0),
      totalEInEE(0.0),
      totalEInEB(0),
      totalEInES(0.0),
      totalEInEEzp(0.0),
      totalEInEEzm(0.0),
      totalEInESzp(0.0),
      totalEInESzm(0.0),
      totalHits(0),
      nHitsInEE(0),
      nHitsInEB(0),
      nHitsInES(0),
      nHitsIn1ES(0),
      nHitsIn2ES(0),
      nCrystalInEB(0),
      nCrystalInEEzp(0),
      nCrystalInEEzm(0),
      nHitsIn1ESzp(0),
      nHitsIn1ESzm(0),
      nHitsIn2ESzp(0),
      nHitsIn2ESzm(0),
      thePID(0),
      label(iPSet.getUntrackedParameter<std::string>("instanceLabel", "EcalValidInfo")) {
  produces<PEcalValidInfo>(label);

  for (int i = 0; i < 26; i++) {
    eBX0[i] = 0.0;
    eEX0[i] = 0.0;
  }
}

EcalSimHitsValidProducer::~EcalSimHitsValidProducer() {}

void EcalSimHitsValidProducer::produce(edm::Event &e, const edm::EventSetup &) {
  std::unique_ptr<PEcalValidInfo> product(new PEcalValidInfo);
  fillEventInfo(*product);
  e.put(std::move(product), label);
}

void EcalSimHitsValidProducer::fillEventInfo(PEcalValidInfo &product) {
  if (ee1 != 0) {
    product.ee1 = ee1;
    product.ee4 = ee4;
    product.ee9 = ee9;
    product.ee16 = ee16;
    product.ee25 = ee25;
    for (int i = 0; i < 26; i++) {
      product.eEX0.push_back(eEX0[i]);
    }
  }

  if (eb1 != 0) {
    product.eb1 = eb1;
    product.eb4 = eb4;
    product.eb9 = eb9;
    product.eb16 = eb16;
    product.eb25 = eb25;
    for (int i = 0; i < 26; i++) {
      product.eBX0.push_back(eBX0[i]);
    }
  }

  product.totalEInEE = totalEInEE;
  product.totalEInEB = totalEInEB;
  product.totalEInES = totalEInES;

  product.totalEInEEzp = totalEInEEzp;
  product.totalEInEEzm = totalEInEEzm;

  product.totalEInESzp = totalEInESzp;
  product.totalEInESzm = totalEInESzm;

  product.totalHits = totalHits;
  product.nHitsInEE = nHitsInEE;
  product.nHitsInEB = nHitsInEB;
  product.nHitsInES = nHitsInES;
  product.nHitsIn1ES = nHitsIn1ES;
  product.nHitsIn2ES = nHitsIn2ES;
  product.nCrystalInEB = nCrystalInEB;
  product.nCrystalInEEzp = nCrystalInEEzp;
  product.nCrystalInEEzm = nCrystalInEEzm;

  product.nHitsIn1ESzp = nHitsIn1ESzp;
  product.nHitsIn1ESzm = nHitsIn1ESzm;
  product.nHitsIn2ESzp = nHitsIn2ESzp;
  product.nHitsIn2ESzm = nHitsIn2ESzm;

  product.eOf1ES = eOf1ES;
  product.eOf2ES = eOf2ES;
  product.zOfES = zOfES;

  product.eOf1ESzp = eOf1ESzp;
  product.eOf1ESzm = eOf1ESzm;
  product.eOf2ESzp = eOf2ESzp;
  product.eOf2ESzm = eOf2ESzm;

  product.phiOfEECaloG4Hit = phiOfEECaloG4Hit;
  product.etaOfEECaloG4Hit = etaOfEECaloG4Hit;
  product.eOfEECaloG4Hit = eOfEECaloG4Hit;
  product.eOfEEPlusCaloG4Hit = eOfEEPlusCaloG4Hit;
  product.eOfEEMinusCaloG4Hit = eOfEEMinusCaloG4Hit;
  product.tOfEECaloG4Hit = tOfEECaloG4Hit;

  product.phiOfESCaloG4Hit = phiOfESCaloG4Hit;
  product.etaOfESCaloG4Hit = etaOfESCaloG4Hit;
  product.eOfESCaloG4Hit = eOfESCaloG4Hit;
  product.tOfESCaloG4Hit = tOfESCaloG4Hit;

  product.phiOfEBCaloG4Hit = phiOfEBCaloG4Hit;
  product.etaOfEBCaloG4Hit = etaOfEBCaloG4Hit;
  product.eOfEBCaloG4Hit = eOfEBCaloG4Hit;
  product.tOfEBCaloG4Hit = tOfEBCaloG4Hit;

  product.theMomentum = theMomentum;
  product.theVertex = theVertex;
  product.thePID = thePID;
}

void EcalSimHitsValidProducer::update(const BeginOfEvent *) {
  ee1 = 0.0;
  ee4 = 0.0;
  ee9 = 0.0;
  ee16 = 0.0;
  ee25 = 0.0;

  eb1 = 0.0;
  eb4 = 0.0;
  eb9 = 0.0;
  eb16 = 0.0;
  eb25 = 0.0;

  totalEInEE = 0.0;
  totalEInEB = 0.0;
  totalEInES = 0.0;

  totalEInEEzp = 0.0;
  totalEInEEzm = 0.0;
  totalEInESzp = 0.0;
  totalEInESzm = 0.0;

  totalHits = 0;
  nHitsInEE = 0;
  nHitsInEB = 0;
  nHitsInES = 0;
  nHitsIn1ES = 0;
  nHitsIn2ES = 0;
  nCrystalInEB = 0;
  nCrystalInEEzp = 0;
  nCrystalInEEzm = 0;

  nHitsIn1ESzp = 0;
  nHitsIn1ESzm = 0;
  nHitsIn2ESzp = 0;
  nHitsIn2ESzm = 0;

  for (int i = 0; i < 26; i++) {
    eBX0[i] = 0.0;
    eEX0[i] = 0.0;
  }

  eOf1ES.clear();
  eOf2ES.clear();
  zOfES.clear();

  eOf1ESzp.clear();
  eOf1ESzm.clear();
  eOf2ESzp.clear();
  eOf2ESzm.clear();

  phiOfEECaloG4Hit.clear();
  etaOfEECaloG4Hit.clear();
  tOfEECaloG4Hit.clear();
  eOfEECaloG4Hit.clear();
  eOfEEPlusCaloG4Hit.clear();
  eOfEEMinusCaloG4Hit.clear();

  phiOfESCaloG4Hit.clear();
  etaOfESCaloG4Hit.clear();
  tOfESCaloG4Hit.clear();
  eOfESCaloG4Hit.clear();

  phiOfEBCaloG4Hit.clear();
  etaOfEBCaloG4Hit.clear();
  tOfEBCaloG4Hit.clear();
  eOfEBCaloG4Hit.clear();
}

void EcalSimHitsValidProducer::update(const EndOfEvent *evt) {
  int trackID = 0;
  G4PrimaryParticle *thePrim = nullptr;
  int nvertex = (*evt)()->GetNumberOfPrimaryVertex();
  if (nvertex <= 0) {
    edm::LogWarning("EcalSimHitsValidProducer") << " No Vertex in this Event!";
  } else {
    for (int i = 0; i < nvertex; i++) {
      G4PrimaryVertex *avertex = (*evt)()->GetPrimaryVertex(i);
      if (avertex == nullptr)
        edm::LogWarning("EcalSimHitsValidProducer") << " Pointer to vertex is NULL!";
      else {
        float x0 = avertex->GetX0();
        float y0 = avertex->GetY0();
        float z0 = avertex->GetZ0();
        float t0 = avertex->GetT0();
        theVertex.SetCoordinates(x0, y0, z0, t0);

        int npart = avertex->GetNumberOfParticle();
        if (npart == 0)
          edm::LogWarning("EcalSimHitsValidProducer") << " No primary particle in this event";
        else {
          if (thePrim == nullptr)
            thePrim = avertex->GetPrimary(trackID);
        }
      }
    }

    // the direction of momentum of primary particles
    double pInit = 0;  // etaInit =0, phiInit =0, // UNUSED
    if (thePrim != nullptr) {
      double px = thePrim->GetPx();
      double py = thePrim->GetPy();
      double pz = thePrim->GetPz();
      theMomentum.SetCoordinates(px, py, pz, 0.);

      pInit = sqrt(pow(px, 2.) + pow(py, 2.) + pow(pz, 2.));
      if (pInit == 0)
        edm::LogWarning("EcalSimHitsValidProducer") << " Primary has p = 0 ; ";
      else {
        theMomentum.SetE(pInit);
        // double costheta  = pz/pInit; // UNUSED
        // double theta = acos(std::min(std::max(costheta, -1.),1.)); // UNUSED
        // etaInit = -log(tan(theta/2)); // UNUSED

        // if ( px != 0 || py != 0) phiInit = atan2(py,px); // UNUSED
      }

      thePID = thePrim->GetPDGcode();
    } else {
      edm::LogWarning("EcalSimHitsValidProducer") << " Could not find the primary particle!!";
    }
  }
  // hit map for EB for matrices
  G4HCofThisEvent *allHC = (*evt)()->GetHCofThisEvent();
  int EBHCid = G4SDManager::GetSDMpointer()->GetCollectionID("EcalHitsEB");
  int EEHCid = G4SDManager::GetSDMpointer()->GetCollectionID("EcalHitsEE");
  int SEHCid = G4SDManager::GetSDMpointer()->GetCollectionID("EcalHitsES");

  CaloG4HitCollection *theEBHC = (CaloG4HitCollection *)allHC->GetHC(EBHCid);
  CaloG4HitCollection *theEEHC = (CaloG4HitCollection *)allHC->GetHC(EEHCid);
  CaloG4HitCollection *theSEHC = (CaloG4HitCollection *)allHC->GetHC(SEHCid);

  nHitsInEE = theEEHC->entries();
  nHitsInEB = theEBHC->entries();
  nHitsInES = theSEHC->entries();
  totalHits = nHitsInEE + nHitsInEB + nHitsInES;

  //   EB Hit collection start
  MapType ebmap;
  int theebhc_entries = theEBHC->entries();
  for (int j = 0; j < theebhc_entries; j++) {
    CaloG4Hit *aHit = (*theEBHC)[j];
    totalEInEB += aHit->getEnergyDeposit();
    float he = aHit->getEnergyDeposit();
    float htime = aHit->getTimeSlice();

    math::XYZPoint hpos = aHit->getEntry();
    float htheta = hpos.theta();
    float heta = -log(tan(htheta * 0.5));
    float hphi = hpos.phi();

    phiOfEBCaloG4Hit.push_back(hphi);
    etaOfEBCaloG4Hit.push_back(heta);
    tOfEBCaloG4Hit.push_back(htime);
    eOfEBCaloG4Hit.push_back(he);
    uint32_t crystid = aHit->getUnitID();
    ebmap[crystid] += aHit->getEnergyDeposit();
  }

  nCrystalInEB = ebmap.size();

  //   EE Hit collection start
  MapType eemap, eezpmap, eezmmap;
  int theeehc_entries = theEEHC->entries();
  for (int j = 0; j < theeehc_entries; j++) {
    CaloG4Hit *aHit = (*theEEHC)[j];
    totalEInEE += aHit->getEnergyDeposit();
    float he = aHit->getEnergyDeposit();
    float htime = aHit->getTimeSlice();

    math::XYZPoint hpos = aHit->getEntry();
    float htheta = hpos.theta();
    float heta = -log(tan(htheta * 0.5));
    float hphi = hpos.phi();
    phiOfEECaloG4Hit.push_back(hphi);
    etaOfEECaloG4Hit.push_back(heta);
    tOfEECaloG4Hit.push_back(htime);
    eOfEECaloG4Hit.push_back(he);

    uint32_t crystid = aHit->getUnitID();
    EEDetId myEEid(crystid);
    if (myEEid.zside() == -1) {
      totalEInEEzm += aHit->getEnergyDeposit();
      eOfEEMinusCaloG4Hit.push_back(he);
      eezmmap[crystid] += aHit->getEnergyDeposit();
    }
    if (myEEid.zside() == 1) {
      totalEInEEzp += aHit->getEnergyDeposit();
      eOfEEPlusCaloG4Hit.push_back(he);
      eezpmap[crystid] += aHit->getEnergyDeposit();
    }

    eemap[crystid] += aHit->getEnergyDeposit();
  }

  nCrystalInEEzm = eezmmap.size();
  nCrystalInEEzp = eezpmap.size();

  // Hits from ES
  int thesehc_entries = theSEHC->entries();
  for (int j = 0; j < thesehc_entries; j++) {
    CaloG4Hit *aHit = (*theSEHC)[j];
    totalEInES += aHit->getEnergyDeposit();
    ESDetId esid = ESDetId(aHit->getUnitID());

    if (esid.zside() == -1) {
      totalEInESzm += aHit->getEnergyDeposit();

      if (esid.plane() == 1) {
        nHitsIn1ESzm++;
        eOf1ESzm.push_back(aHit->getEnergyDeposit());
      } else if (esid.plane() == 2) {
        nHitsIn2ESzm++;
        eOf2ESzm.push_back(aHit->getEnergyDeposit());
      }
    }
    if (esid.zside() == 1) {
      totalEInESzp += aHit->getEnergyDeposit();

      if (esid.plane() == 1) {
        nHitsIn1ESzp++;
        eOf1ESzp.push_back(aHit->getEnergyDeposit());
      } else if (esid.plane() == 2) {
        nHitsIn2ESzp++;
        eOf2ESzp.push_back(aHit->getEnergyDeposit());
      }
    }
  }

  uint32_t eemaxid = getUnitWithMaxEnergy(eemap);
  uint32_t ebmaxid = getUnitWithMaxEnergy(ebmap);
  if (eemap[eemaxid] > ebmap[ebmaxid]) {
    uint32_t centerid = getUnitWithMaxEnergy(eemap);
    EEDetId myEEid(centerid);
    int ix = myEEid.ix();
    int iy = myEEid.iy();
    int iz = myEEid.zside();

    ee1 = energyInEEMatrix(1, 1, ix, iy, iz, eemap);
    ee9 = energyInEEMatrix(3, 3, ix, iy, iz, eemap);
    ee25 = energyInEEMatrix(5, 5, ix, iy, iz, eemap);
    MapType neweemap;
    if (fillEEMatrix(3, 3, ix, iy, iz, neweemap, eemap)) {
      ee4 = eCluster2x2(neweemap);
    }
    if (fillEEMatrix(5, 5, ix, iy, iz, neweemap, eemap)) {
      ee16 = eCluster4x4(ee9, neweemap);
    }
  } else {
    uint32_t ebcenterid = getUnitWithMaxEnergy(ebmap);
    EBDetId myEBid(ebcenterid);
    int bx = myEBid.ietaAbs();
    int by = myEBid.iphi();
    int bz = myEBid.zside();
    eb1 = energyInEBMatrix(1, 1, bx, by, bz, ebmap);
    eb9 = energyInEBMatrix(3, 3, bx, by, bz, ebmap);
    eb25 = energyInEBMatrix(5, 5, bx, by, bz, ebmap);

    MapType newebmap;
    if (fillEBMatrix(3, 3, bx, by, bz, newebmap, ebmap)) {
      eb4 = eCluster2x2(newebmap);
    }
    if (fillEBMatrix(5, 5, bx, by, bz, newebmap, ebmap)) {
      eb16 = eCluster4x4(eb9, newebmap);
    }
  }
}

void EcalSimHitsValidProducer::update(const G4Step *aStep) {
  G4StepPoint *preStepPoint = aStep->GetPreStepPoint();
  const G4ThreeVector &hitPoint = preStepPoint->GetPosition();
  G4VPhysicalVolume *currentPV = preStepPoint->GetPhysicalVolume();
  const G4String &name = currentPV->GetName();
  std::string crystal;
  crystal.assign(name, 0, 4);

  float Edeposit = aStep->GetTotalEnergyDeposit();
  if (crystal == "EFRY" && Edeposit > 0.0) {
    float z = hitPoint.z();
    float detz = fabs(fabs(z) - 3200);
    int x0 = (int)floor(detz / 8.9);
    if (x0 < 26) {
      eEX0[x0] += Edeposit;
    }
  }
  if (crystal == "EBRY" && Edeposit > 0.0) {
    float x = hitPoint.x();
    float y = hitPoint.y();
    float r = sqrt(x * x + y * y);
    float detr = r - 1290;
    int x0 = (int)floor(detr / 8.9);
    eBX0[x0] += Edeposit;
  }
}

bool EcalSimHitsValidProducer::fillEEMatrix(
    int nCellInEta, int nCellInPhi, int CentralEta, int CentralPhi, int CentralZ, MapType &fillmap, MapType &themap) {
  int goBackInEta = nCellInEta / 2;
  int goBackInPhi = nCellInPhi / 2;

  int startEta = CentralEta - goBackInEta;
  int startPhi = CentralPhi - goBackInPhi;

  int i = 0;
  for (int ieta = startEta; ieta < startEta + nCellInEta; ieta++) {
    for (int iphi = startPhi; iphi < startPhi + nCellInPhi; iphi++) {
      uint32_t index;

      if (EEDetId::validDetId(ieta, iphi, CentralZ)) {
        index = EEDetId(ieta, iphi, CentralZ).rawId();
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

bool EcalSimHitsValidProducer::fillEBMatrix(
    int nCellInEta, int nCellInPhi, int CentralEta, int CentralPhi, int CentralZ, MapType &fillmap, MapType &themap) {
  int goBackInEta = nCellInEta / 2;
  int goBackInPhi = nCellInPhi / 2;

  int startEta = CentralZ * CentralEta - goBackInEta;
  int startPhi = CentralPhi - goBackInPhi;

  int i = 0;
  for (int ieta = startEta; ieta < startEta + nCellInEta; ieta++) {
    for (int iphi = startPhi; iphi < startPhi + nCellInPhi; iphi++) {
      uint32_t index;
      if (abs(ieta) > 85 || abs(ieta) < 1) {
        continue;
      }
      if (iphi < 1) {
        index = EBDetId(ieta, iphi + 360).rawId();
      } else if (iphi > 360) {
        index = EBDetId(ieta, iphi - 360).rawId();
      } else {
        index = EBDetId(ieta, iphi).rawId();
      }
      fillmap[i++] = themap[index];
    }
  }

  uint32_t ebcenterid = getUnitWithMaxEnergy(themap);

  if (fillmap[i / 2] == themap[ebcenterid])
    return true;
  else
    return false;
}

float EcalSimHitsValidProducer::eCluster2x2(MapType &themap) {
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

float EcalSimHitsValidProducer::eCluster4x4(float e33, MapType &themap) {
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

float EcalSimHitsValidProducer::energyInEEMatrix(
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

      LogDebug("EcalSimHitsValidProducer")
          << " EnergyInEEMatrix: ix - iy - E = " << ix << "  " << iy << " " << themap[index];
    }
  }

  LogDebug("EcalSimHitsValidProducer") << " EnergyInEEMatrix: energy in " << nCellInX << " cells in x times "
                                       << nCellInY << " cells in y matrix = " << totalEnergy << " for " << ncristals
                                       << " crystals";

  return totalEnergy;
}

float EcalSimHitsValidProducer::energyInEBMatrix(
    int nCellInEta, int nCellInPhi, int centralEta, int centralPhi, int centralZ, MapType &themap) {
  int ncristals = 0;
  float totalEnergy = 0.;

  int goBackInEta = nCellInEta / 2;
  int goBackInPhi = nCellInPhi / 2;
  int startEta = centralZ * centralEta - goBackInEta;
  int startPhi = centralPhi - goBackInPhi;

  for (int ieta = startEta; ieta < startEta + nCellInEta; ieta++) {
    for (int iphi = startPhi; iphi < startPhi + nCellInPhi; iphi++) {
      uint32_t index;
      if (abs(ieta) > 85 || abs(ieta) < 1) {
        continue;
      }
      if (iphi < 1) {
        index = EBDetId(ieta, iphi + 360).rawId();
      } else if (iphi > 360) {
        index = EBDetId(ieta, iphi - 360).rawId();
      } else {
        index = EBDetId(ieta, iphi).rawId();
      }

      totalEnergy += themap[index];
      ncristals += 1;

      LogDebug("EcalSimHitsValidProducer")
          << " EnergyInEBMatrix: ieta - iphi - E = " << ieta << "  " << iphi << " " << themap[index];
    }
  }

  LogDebug("EcalSimHitsValidProducer") << " EnergyInEBMatrix: energy in " << nCellInEta << " cells in eta times "
                                       << nCellInPhi << " cells in phi matrix = " << totalEnergy << " for " << ncristals
                                       << " crystals";

  return totalEnergy;
}

uint32_t EcalSimHitsValidProducer::getUnitWithMaxEnergy(MapType &themap) {
  uint32_t unitWithMaxEnergy = 0;
  float maxEnergy = 0.;

  MapType::iterator iter;
  for (iter = themap.begin(); iter != themap.end(); iter++) {
    if (maxEnergy < (*iter).second) {
      maxEnergy = (*iter).second;
      unitWithMaxEnergy = (*iter).first;
    }
  }

  LogDebug("EcalSimHitsValidProducer") << " Max energy of " << maxEnergy << " MeV was found in Unit id 0x" << std::hex
                                       << unitWithMaxEnergy << std::dec;

  return unitWithMaxEnergy;
}

#include "SimG4Core/Notification/interface/BeginOfJob.h"
#include "SimG4Core/Notification/interface/BeginOfRun.h"
#include "SimG4Core/Notification/interface/BeginOfEvent.h"
#include "SimG4Core/Notification/interface/EndOfEvent.h"

// to retreive hits
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "SimG4CMS/Calo/interface/HcalTestAnalysis.h"
#include "SimG4CMS/Calo/interface/HCalSD.h"
#include "SimG4CMS/Calo/interface/CaloG4Hit.h"
#include "SimG4CMS/Calo/interface/CaloG4HitCollection.h"
#include "DataFormats/Math/interface/Point3D.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/Records/interface/HcalSimNumberingRecord.h"

#include "G4SDManager.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4VProcess.hh"
#include "G4HCofThisEvent.hh"
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "Randomize.hh"

#include <cmath>
#include <iostream>
#include <iomanip>

HcalTestAnalysis::HcalTestAnalysis(const edm::ParameterSet& p)
    : addTower_(3), tuples_(nullptr), hcons_(nullptr), org_(nullptr) {
  edm::ParameterSet m_Anal = p.getParameter<edm::ParameterSet>("HcalTestAnalysis");
  eta0_ = m_Anal.getParameter<double>("Eta0");
  phi0_ = m_Anal.getParameter<double>("Phi0");
  int laygroup = m_Anal.getParameter<int>("LayerGrouping");
  centralTower_ = m_Anal.getParameter<int>("CentralTower");
  names_ = m_Anal.getParameter<std::vector<std::string> >("Names");
  fileName_ = m_Anal.getParameter<std::string>("FileName");

  tuplesManager_.reset(nullptr);
  numberingFromDDD_.reset(nullptr);
  edm::LogVerbatim("HcalSim") << "HcalTestAnalysis:: Initialised as observer of begin/end events"
                              << " and of G4step";

  count_ = 0;
  group_ = layerGrouping(laygroup);
  nGroup_ = 0;
  for (unsigned int i = 0; i < group_.size(); i++)
    if (group_[i] > nGroup_)
      nGroup_ = group_[i];
  tower_ = towersToAdd(centralTower_, addTower_);
  nTower_ = tower_.size() / 2;

  edm::LogVerbatim("HcalSim") << "HcalTestAnalysis:: initialised for " << nGroup_ << " Longitudinal groups and "
                              << nTower_ << " towers";

  // qie
  myqie_.reset(new HcalQie(p));
}

HcalTestAnalysis::~HcalTestAnalysis() {
  edm::LogVerbatim("HcalSim") << "HcalTestAnalysis: -------->  Total number of selected entries : " << count_;
  edm::LogVerbatim("HcalSim") << "HcalTestAnalysis: Pointers:: HistoClass " << tuples_ << ", Numbering Scheme " << org_;
}

std::vector<int> HcalTestAnalysis::layerGrouping(int group) {
  std::vector<int> temp(19);
  if (group <= 1) {
    int grp[19] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2};
    for (int i = 0; i < 19; i++)
      temp[i] = grp[i];
  } else if (group == 2) {
    int grp[19] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
    for (int i = 0; i < 19; i++)
      temp[i] = grp[i];
  } else if (group == 3) {
    int grp[19] = {1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7};
    for (int i = 0; i < 19; i++)
      temp[i] = grp[i];
  } else if (group == 4) {
    int grp[19] = {1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7};
    for (int i = 0; i < 19; i++)
      temp[i] = grp[i];
  } else {
    int grp[19] = {1, 1, 2, 3, 3, 4, 4, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 7, 7};
    for (int i = 0; i < 19; i++)
      temp[i] = grp[i];
  }

  edm::LogVerbatim("HcalSim") << "HcalTestAnalysis:: Layer Grouping ";
  for (int i = 0; i < 19; i++)
    edm::LogVerbatim("HcalSim") << "HcalTestAnalysis: Group[" << i << "] = " << temp[i];
  return temp;
}

std::vector<int> HcalTestAnalysis::towersToAdd(int centre, int nadd) {
  int etac = (centre / 100) % 100;
  int phic = (centre % 100);
  int etamin, etamax, phimin, phimax;
  if (etac > 0) {
    etamin = etac - nadd;
    etamax = etac + nadd;
  } else {
    etamin = etac;
    etamax = etac;
  }
  if (phic > 0) {
    phimin = phic - nadd;
    phimax = phic + nadd;
  } else {
    phimin = phic;
    phimax = phic;
  }

  int nbuf, kount = 0;
  nbuf = (etamax - etamin + 1) * (phimax - phimin + 1);
  std::vector<int> temp(2 * nbuf);
  for (int eta = etamin; eta <= etamax; eta++) {
    for (int phi = phimin; phi <= phimax; phi++) {
      temp[kount] = (eta * 100 + phi);
      temp[kount + nbuf] = std::max(abs(eta - etac), abs(phi - phic));
      kount++;
    }
  }

  edm::LogVerbatim("HcalSim") << "HcalTestAnalysis:: Towers to be considered for Central " << centre << " and " << nadd
                              << " on either side";
  for (int i = 0; i < nbuf; i++)
    edm::LogVerbatim("HcalSim") << "HcalTestAnalysis: Tower[" << std::setw(3) << i << "] " << temp[i] << " "
                                << temp[nbuf + i];
  return temp;
}

//==================================================================== per JOB

void HcalTestAnalysis::update(const BeginOfJob* job) {
  // Numbering From DDD
  edm::ESHandle<HcalDDDSimConstants> hdc;
  (*job)()->get<HcalSimNumberingRecord>().get(hdc);
  hcons_ = hdc.product();
  edm::LogVerbatim("HcalSim") << "HcalTestAnalysis:: Initialise HcalNumberingFromDDD for " << names_[0];
  numberingFromDDD_.reset(new HcalNumberingFromDDD(hcons_));

  // Ntuples
  tuplesManager_.reset(new HcalTestHistoManager(fileName_));

  // Numbering scheme
  org_ = new HcalTestNumberingScheme(false);
}

//==================================================================== per RUN
void HcalTestAnalysis::update(const BeginOfRun* run) {
  int irun = (*run)()->GetRunID();
  edm::LogVerbatim("HcalSim") << "HcalTestAnalysis:: Begin of Run = " << irun;

  bool loop = true, eta = true, phi = true;
  int etac = (centralTower_ / 100) % 100;
  if (etac == 0) {
    etac = 1;
    eta = false;
  }
  int phic = (centralTower_ % 100);
  if (phic == 0) {
    phic = 1;
    phi = false;
  }
  int idet = static_cast<int>(HcalBarrel);
  while (loop) {
    HcalCellType::HcalCell tmp = hcons_->cell(idet, 1, 1, etac, phic);
    if (tmp.ok) {
      if (eta)
        eta0_ = tmp.eta;
      if (phi)
        phi0_ = tmp.phi;
      loop = false;
    } else if (idet == static_cast<int>(HcalBarrel)) {
      idet = static_cast<int>(HcalEndcap);
    } else if (idet == static_cast<int>(HcalEndcap)) {
      idet = static_cast<int>(HcalForward);
    } else {
      loop = false;
    }
  }

  edm::LogVerbatim("HcalSim") << "HcalTestAnalysis:: Central Tower " << centralTower_
                              << " corresponds to eta0 = " << eta0_ << " phi0 = " << phi0_;

  std::string sdname = names_[0];
  G4SDManager* sd = G4SDManager::GetSDMpointerIfExist();
  if (sd != nullptr) {
    G4VSensitiveDetector* aSD = sd->FindSensitiveDetector(sdname);
    if (aSD == nullptr) {
      edm::LogWarning("HcalSim") << "HcalTestAnalysis::beginOfRun: No SD with "
                                 << "name " << sdname << " in this Setup";
    } else {
      HCalSD* theCaloSD = dynamic_cast<HCalSD*>(aSD);
      edm::LogVerbatim("HcalSim") << "HcalTestAnalysis::beginOfRun: Finds SD with name " << theCaloSD->GetName()
                                  << " in this Setup";
      if (org_) {
        theCaloSD->setNumberingScheme(org_);
        edm::LogVerbatim("HcalSim") << "HcalTestAnalysis::beginOfRun: set a new numbering scheme";
      }
    }
  } else {
    edm::LogWarning("HcalSim") << "HcalTestAnalysis::beginOfRun: Could not get"
                               << " SD Manager!";
  }
}

//=================================================================== per EVENT
void HcalTestAnalysis::update(const BeginOfEvent* evt) {
  // create tuple object
  tuples_ = new HcalTestHistoClass();
  // Reset counters
  tuples_->setCounters();

  int i = 0;
  edepEB_ = edepEE_ = edepHB_ = edepHE_ = edepHO_ = 0.;
  for (i = 0; i < 20; i++)
    edepl_[i] = 0.;
  for (i = 0; i < 20; i++)
    mudist_[i] = -1.;

  edm::LogVerbatim("HcalSim") << "HcalTestAnalysis: Begin of event = " << (*evt)()->GetEventID();
}

//=================================================================== each STEP
void HcalTestAnalysis::update(const G4Step* aStep) {
  if (aStep != nullptr) {
    G4VPhysicalVolume* curPV = aStep->GetPreStepPoint()->GetPhysicalVolume();
    G4String name = curPV->GetName();
    name.assign(name, 0, 3);
    double edeposit = aStep->GetTotalEnergyDeposit();
    int layer = -1;
    if (name == "EBR") {
      edepEB_ += edeposit;
    } else if (name == "EFR") {
      edepEE_ += edeposit;
    } else if (name == "HBS") {
      layer = (curPV->GetCopyNo() / 10) % 100;
      if (layer >= 0 && layer < 17) {
        edepHB_ += edeposit;
      } else {
        edm::LogWarning("HcalSim") << "HcalTestAnalysis::Error in HB " << curPV->GetName() << curPV->GetCopyNo();
        layer = -1;
      }
    } else if (name == "HES") {
      layer = (curPV->GetCopyNo() / 10) % 100;
      if (layer >= 0 && layer < 19) {
        edepHE_ += edeposit;
      } else {
        edm::LogWarning("HcalSim") << "HcalTestAnalysis::Error in HE " << curPV->GetName() << curPV->GetCopyNo();
        layer = -1;
      }
    } else if (name == "HTS") {
      layer = (curPV->GetCopyNo() / 10) % 100;
      if (layer >= 17 && layer < 20) {
        edepHO_ += edeposit;
      } else {
        edm::LogWarning("HcalSim") << "HcalTestAnalysis::Error in HO " << curPV->GetName() << curPV->GetCopyNo();
        layer = -1;
      }
    }
    if (layer >= 0 && layer < 20) {
      edepl_[layer] += edeposit;

      // Calculate the distance if it is a muon
      G4String part = aStep->GetTrack()->GetDefinition()->GetParticleName();
      if ((part == "mu-" || part == "mu+") && mudist_[layer] < 0) {
        math::XYZPoint pos(aStep->GetPreStepPoint()->GetPosition().x(),
                           aStep->GetPreStepPoint()->GetPosition().y(),
                           aStep->GetPreStepPoint()->GetPosition().z());
        double theta = pos.theta();
        double eta = -log(tan(theta * 0.5));
        double phi = pos.phi();
        double dist = sqrt((eta - eta0_) * (eta - eta0_) + (phi - phi0_) * (phi - phi0_));
        mudist_[layer] = dist * std::sqrt(pos.perp2());
      }
    }

    if (layer >= 0 && layer < 20) {
      edm::LogVerbatim("HcalSim") << "HcalTestAnalysis:: G4Step: " << name << " Layer " << std::setw(3) << layer
                                  << " Edep " << std::setw(6) << edeposit / MeV << " MeV";
    }
  } else {
    edm::LogVerbatim("HcalSim") << "HcalTestAnalysis:: G4Step: Null Step";
  }
}

//================================================================ End of EVENT
void HcalTestAnalysis::update(const EndOfEvent* evt) {
  ++count_;
  // Fill event input
  fill(evt);
  edm::LogVerbatim("HcalSim") << "HcalTestAnalysis:: ---  after Fill";

  // Qie analysis
  CLHEP::HepRandomEngine* engine = G4Random::getTheEngine();
  qieAnalysis(engine);
  edm::LogVerbatim("HcalSim") << "HcalTestAnalysis:: ---  after QieAnalysis";

  // Layers tuples filling
  layerAnalysis();
  edm::LogVerbatim("HcalSim") << "HcalTestAnalysis:: ---  after LayerAnalysis";

  // Writing the data to the Tree
  tuplesManager_.get()->fillTree(tuples_);  // (no need to delete it...)
  tuples_ = nullptr;                        // but avoid to reuse it...
  edm::LogVerbatim("HcalSim") << "HcalTestAnalysis:: --- after fillTree";
}

//---------------------------------------------------
void HcalTestAnalysis::fill(const EndOfEvent* evt) {
  edm::LogVerbatim("HcalSim") << "HcalTestAnalysis: Fill event " << (*evt)()->GetEventID();

  // access to the G4 hit collections
  G4HCofThisEvent* allHC = (*evt)()->GetHCofThisEvent();

  int nhc = 0, neb = 0, nef = 0, j = 0;
  caloHitCache_.erase(caloHitCache_.begin(), caloHitCache_.end());

  // Hcal
  int HCHCid = G4SDManager::GetSDMpointer()->GetCollectionID(names_[0]);
  CaloG4HitCollection* theHCHC = (CaloG4HitCollection*)allHC->GetHC(HCHCid);
  edm::LogVerbatim("HcalSim") << "HcalTestAnalysis :: Hit Collection for " << names_[0] << " of ID " << HCHCid
                              << " is obtained at " << theHCHC;
  int hchc_entries = theHCHC->entries();
  if (HCHCid >= 0 && theHCHC != nullptr) {
    for (j = 0; j < hchc_entries; j++) {
      CaloG4Hit* aHit = (*theHCHC)[j];

      double e = aHit->getEnergyDeposit() / GeV;
      double time = aHit->getTimeSlice();

      math::XYZPoint pos = aHit->getPosition();
      double theta = pos.theta();
      double eta = -log(tan(theta * 0.5));
      double phi = pos.phi();

      uint32_t unitID = aHit->getUnitID();
      int subdet, zside, layer, etaIndex, phiIndex, lay;
      org_->unpackHcalIndex(unitID, subdet, zside, layer, etaIndex, phiIndex, lay);
      double jitter = time - timeOfFlight(subdet, lay, eta);
      if (jitter < 0)
        jitter = 0;
      CaloHit hit(subdet, lay, e, eta, phi, jitter, unitID);
      caloHitCache_.push_back(hit);
      nhc++;

      std::string det = "HB";
      if (subdet == static_cast<int>(HcalForward)) {
        det = "HF";
      } else if (subdet == static_cast<int>(HcalEndcap)) {
        if (etaIndex <= 20) {
          det = "HES";
        } else {
          det = "HED";
        }
      }

      edm::LogVerbatim("HcalSim") << "HcalTest: " << det << "  layer " << std::setw(2) << layer << " time "
                                  << std::setw(6) << time << " theta " << std::setw(8) << theta << " eta "
                                  << std::setw(8) << eta << " phi " << std::setw(8) << phi << " e " << std::setw(8)
                                  << e;
    }
  }
  edm::LogVerbatim("HcalSim") << "HcalTestAnalysis::HCAL hits : " << nhc;

  // EB
  int EBHCid = G4SDManager::GetSDMpointer()->GetCollectionID(names_[1]);
  CaloG4HitCollection* theEBHC = (CaloG4HitCollection*)allHC->GetHC(EBHCid);
  edm::LogVerbatim("HcalSim") << "HcalTestAnalysis :: Hit Collection for " << names_[1] << " of ID " << EBHCid
                              << " is obtained at " << theEBHC;
  int ebhc_entries = theEBHC->entries();
  if (EBHCid >= 0 && theEBHC != nullptr) {
    for (j = 0; j < ebhc_entries; j++) {
      CaloG4Hit* aHit = (*theEBHC)[j];

      double e = aHit->getEnergyDeposit() / GeV;
      double time = aHit->getTimeSlice();
      std::string det = "EB";

      math::XYZPoint pos = aHit->getPosition();
      double theta = pos.theta();
      double eta = -log(tan(theta / 2.));
      double phi = pos.phi();

      HcalNumberingFromDDD::HcalID id = numberingFromDDD_->unitID(eta, phi, 1, 1);
      uint32_t unitID = org_->getUnitID(id);
      int subdet, zside, layer, ieta, iphi, lay;
      org_->unpackHcalIndex(unitID, subdet, zside, layer, ieta, iphi, lay);
      subdet = 10;
      layer = 0;
      unitID = org_->packHcalIndex(subdet, zside, layer, ieta, iphi, lay);
      CaloHit hit(subdet, lay, e, eta, phi, time, unitID);
      caloHitCache_.push_back(hit);
      neb++;
      edm::LogVerbatim("HcalSim") << "HcalTest: " << det << "  layer " << std::setw(2) << layer << " time "
                                  << std::setw(6) << time << " theta " << std::setw(8) << theta << " eta "
                                  << std::setw(8) << eta << " phi " << std::setw(8) << phi << " e " << std::setw(8)
                                  << e;
    }
  }
  edm::LogVerbatim("HcalSim") << "HcalTestAnalysis::EB hits : " << neb;

  // EE
  int EEHCid = G4SDManager::GetSDMpointer()->GetCollectionID(names_[2]);
  CaloG4HitCollection* theEEHC = (CaloG4HitCollection*)allHC->GetHC(EEHCid);
  edm::LogVerbatim("HcalSim") << "HcalTestAnalysis :: Hit Collection for " << names_[2] << " of ID " << EEHCid
                              << " is obtained at " << theEEHC;
  int eehc_entries = theEEHC->entries();
  if (EEHCid >= 0 && theEEHC != nullptr) {
    for (j = 0; j < eehc_entries; j++) {
      CaloG4Hit* aHit = (*theEEHC)[j];

      double e = aHit->getEnergyDeposit() / GeV;
      double time = aHit->getTimeSlice();
      std::string det = "EE";

      math::XYZPoint pos = aHit->getPosition();
      double theta = pos.theta();
      double eta = -log(tan(theta / 2.));
      double phi = pos.phi();

      HcalNumberingFromDDD::HcalID id = numberingFromDDD_->unitID(eta, phi, 1, 1);
      uint32_t unitID = org_->getUnitID(id);
      int subdet, zside, layer, ieta, iphi, lay;
      org_->unpackHcalIndex(unitID, subdet, zside, layer, ieta, iphi, lay);
      subdet = 11;
      layer = 0;
      unitID = org_->packHcalIndex(subdet, zside, layer, ieta, iphi, lay);
      CaloHit hit(subdet, lay, e, eta, phi, time, unitID);
      caloHitCache_.push_back(hit);
      nef++;
      edm::LogVerbatim("HcalSim") << "HcalTest: " << det << "  layer " << std::setw(2) << layer << " time "
                                  << std::setw(6) << time << " theta " << std::setw(8) << theta << " eta "
                                  << std::setw(8) << eta << " phi " << std::setw(8) << phi << " e " << std::setw(8)
                                  << e;
    }
  }
  edm::LogVerbatim("HcalSim") << "HcalTestAnalysis::EE hits : " << nef;
}

//-----------------------------------------------------------------------------
void HcalTestAnalysis::qieAnalysis(CLHEP::HepRandomEngine* engine) {
  //Fill tuple with hit information
  int hittot = caloHitCache_.size();
  tuples_->fillHits(caloHitCache_);

  //Get the index of the central tower
  HcalNumberingFromDDD::HcalID id = numberingFromDDD_->unitID(eta0_, phi0_, 1, 1);
  uint32_t unitID = org_->getUnitID(id);
  int subdet, zside, layer, ieta, iphi, lay;
  org_->unpackHcalIndex(unitID, subdet, zside, layer, ieta, iphi, lay);
  int laymax = 0;
  std::string det = "Unknown";
  if (subdet == static_cast<int>(HcalBarrel)) {
    laymax = 4;
    det = "HB";
  } else if (subdet == static_cast<int>(HcalEndcap)) {
    laymax = 2;
    det = "HES";
  }
  edm::LogVerbatim("HcalSim") << "HcalTestAnalysis::Qie: " << det << " Eta " << ieta << " Phi " << iphi << " Laymax "
                              << laymax << " Hits " << hittot;

  if (laymax > 0 && hittot > 0) {
    std::vector<CaloHit> hits(hittot);
    std::vector<double> eqielay(80, 0.0), esimlay(80, 0.0), esimtot(4, 0.0);
    std::vector<double> eqietow(200, 0.0), esimtow(200, 0.0), eqietot(4, 0.0);
    int etac = (centralTower_ / 100) % 100;
    int phic = (centralTower_ % 100);

    for (int layr = 0; layr < nGroup_; layr++) {
      /*
      int layx, layy=20;
      for (int i=0; i<20; i++) 
	if (group_[i] == layr+1 && i < layy) layy = i+1;
      if (subdet == static_cast<int>(HcalBarrel)) {
	if      (layy < 2)    layx = 0;
	else if (layy < 17)   layx = 1;
	else if (layy == 17)  layx = 2;
	else                  layx = 3;
      } else {
	if      (layy < 2)    layx = 0;
	else                  layx = 1;
      }
      */
      for (int it = 0; it < nTower_; it++) {
        int nhit = 0;
        double esim = 0;
        for (int k1 = 0; k1 < hittot; k1++) {
          CaloHit hit = caloHitCache_[k1];
          int subdetc = hit.det();
          int layer = hit.layer();
          int group = 0;
          if (layer > 0 && layer < 20)
            group = group_[layer];
          if (subdetc == subdet && group == layr + 1) {
            int zsidec, ietac, iphic, idx;
            unitID = hit.id();
            org_->unpackHcalIndex(unitID, subdetc, zsidec, layer, ietac, iphic, lay);
            if (etac > 0 && phic > 0) {
              idx = ietac * 100 + iphic;
            } else if (etac > 0) {
              idx = ietac * 100;
            } else if (phic > 0) {
              idx = iphic;
            } else {
              idx = 0;
            }
            if (zsidec == zside && idx == tower_[it]) {
              hits[nhit] = hit;
              edm::LogVerbatim("HcalSim") << "HcalTest: Hit " << nhit << " " << hit;
              nhit++;
              esim += hit.e();
            }
          }
        }

        std::vector<int> cd = myqie_->getCode(nhit, hits, engine);
        double eqie = myqie_->getEnergy(cd);

        edm::LogVerbatim("HcalSim") << "HcalTestAnalysis::Qie: Energy in layer " << layr << " Sim " << esim
                                    << " After QIE " << eqie;
        for (int i = 0; i < 4; i++) {
          if (tower_[nTower_ + it] <= i) {
            esimtot[i] += esim;
            eqietot[i] += eqie;
            esimlay[20 * i + layr] += esim;
            eqielay[20 * i + layr] += eqie;
            esimtow[50 * i + it] += esim;
            eqietow[50 * i + it] += eqie;
          }
        }
      }
    }
    edm::LogVerbatim("HcalSim") << "HcalTestAnalysis::Qie: Total energy " << esimtot[3] << " (SimHit) " << eqietot[3]
                                << " (After QIE)";

    std::vector<double> latphi(10);
    int nt = 2 * addTower_ + 1;
    for (int it = 0; it < nt; it++)
      latphi[it] = it - addTower_;
    for (int i = 0; i < 4; i++) {
      double scals = 1, scalq = 1;
      std::vector<double> latfs(10, 0.), latfq(10, 0.), longs(20), longq(20);
      if (esimtot[i] > 0)
        scals = 1. / esimtot[i];
      if (eqietot[i] > 0)
        scalq = 1. / eqietot[i];
      for (int it = 0; it < nTower_; it++) {
        int phib = it % nt;
        latfs[phib] += scals * esimtow[50 * i + it];
        latfq[phib] += scalq * eqietow[50 * i + it];
      }
      for (int layr = 0; layr <= nGroup_; layr++) {
        longs[layr] = scals * esimlay[20 * i + layr];
        longq[layr] = scalq * eqielay[20 * i + layr];
      }
      tuples_->fillQie(i, esimtot[i], eqietot[i], nGroup_, longs, longq, nt, latphi, latfs, latfq);
    }
  }
}

//---------------------------------------------------
void HcalTestAnalysis::layerAnalysis() {
  int i = 0;
  edm::LogVerbatim("HcalSim") << "\n ===>>> HcalTestAnalysis: Energy deposit "
                              << "\n at EB : " << std::setw(6) << edepEB_ / MeV << "\n at EE : " << std::setw(6)
                              << edepEE_ / MeV << "\n at HB : " << std::setw(6) << edepHB_ / MeV
                              << "\n at HE : " << std::setw(6) << edepHE_ / MeV << "\n at HO : " << std::setw(6)
                              << edepHO_ / MeV << "\n ---- HcalTestAnalysis: Energy deposit in Layers";
  for (i = 0; i < 20; i++)
    edm::LogVerbatim("HcalSim") << " Layer " << std::setw(2) << i << " E " << std::setw(8) << edepl_[i] / MeV << " MeV";

  tuples_->fillLayers(edepl_, edepHO_, edepHB_ + edepHE_, mudist_);
}

//---------------------------------------------------
double HcalTestAnalysis::timeOfFlight(int det, int layer, double eta) {
  double theta = 2.0 * atan(exp(-eta));
  double dist = 0.;
  if (det == static_cast<int>(HcalBarrel)) {
    const double rLay[19] = {1836.0,
                             1902.0,
                             1962.0,
                             2022.0,
                             2082.0,
                             2142.0,
                             2202.0,
                             2262.0,
                             2322.0,
                             2382.0,
                             2448.0,
                             2514.0,
                             2580.0,
                             2646.0,
                             2712.0,
                             2776.0,
                             2862.5,
                             3847.0,
                             4052.0};
    if (layer > 0 && layer < 20)
      dist += rLay[layer - 1] * mm / sin(theta);
  } else {
    const double zLay[19] = {4034.0,
                             4032.0,
                             4123.0,
                             4210.0,
                             4297.0,
                             4384.0,
                             4471.0,
                             4558.0,
                             4645.0,
                             4732.0,
                             4819.0,
                             4906.0,
                             4993.0,
                             5080.0,
                             5167.0,
                             5254.0,
                             5341.0,
                             5428.0,
                             5515.0};
    if (layer > 0 && layer < 20)
      dist += zLay[layer - 1] * mm / cos(theta);
  }
  double tmp = dist / c_light / ns;
  edm::LogVerbatim("HcalSim") << "HcalTestAnalysis::timeOfFlight " << tmp << " for det/lay " << det << " " << layer
                              << " eta/theta " << eta << " " << theta / deg << " dist " << dist;
  return tmp;
}

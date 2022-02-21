// -*- C++ -*-
//
// Package:     HcalTestBeam
// Class  :     HcalTB06Analysis
//
// Implementation:
//     Main analysis class for Hcal Test Beam 2006 Analysis
//
// Original Author:
//         Created:  19 November 2015
//

// user include files
#include "SimG4CMS/HcalTestBeam/interface/HcalTB06Histo.h"
#include "SimG4CMS/HcalTestBeam/interface/HcalTB06BeamSD.h"

// to retreive hits
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/Common/interface/Handle.h"

#include "SimG4CMS/Calo/interface/CaloG4Hit.h"
#include "SimG4CMS/Calo/interface/CaloG4HitCollection.h"

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "SimDataFormats/HcalTestBeam/interface/HcalTestBeamNumbering.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"

#include "Randomize.hh"
#include "globals.hh"

#include <CLHEP/Units/GlobalSystemOfUnits.h>
#include <CLHEP/Units/GlobalPhysicalConstants.h>

// system include files
#include <memory>
#include <string>
#include <vector>

//#define EDM_ML_DEBUG

class HcalTB06Analysis : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit HcalTB06Analysis(const edm::ParameterSet& p);
  ~HcalTB06Analysis() override = default;

  void beginJob() override;
  void endJob() override;
  void analyze(const edm::Event& e, const edm::EventSetup& c) override;

  HcalTB06Analysis(const HcalTB06Analysis&) = delete;
  const HcalTB06Analysis& operator=(const HcalTB06Analysis&) = delete;

private:
  const bool m_ECAL;
  const double m_eta;
  const double m_phi;
  const double m_ener;
  const std::vector<int> m_PDG;

  edm::EDGetTokenT<edm::PCaloHitContainer> m_EcalToken;
  const edm::EDGetTokenT<edm::PCaloHitContainer> m_HcalToken;
  const edm::EDGetTokenT<edm::PCaloHitContainer> m_BeamToken;

  const edm::ParameterSet m_ptb;
  const double m_timeLimit;
  const double m_widthEcal;
  const double m_widthHcal;
  const double m_factEcal;
  const double m_factHcal;
  const double m_eMIP;

  int count;
  int m_idxetaEcal;
  int m_idxphiEcal;
  int m_idxetaHcal;
  int m_idxphiHcal;

  std::unique_ptr<HcalTB06Histo> m_histo;
};

HcalTB06Analysis::HcalTB06Analysis(const edm::ParameterSet& p)
    : m_ECAL(p.getParameter<bool>("ECAL")),
      m_eta(p.getParameter<double>("MinEta")),
      m_phi(p.getParameter<double>("MinPhi")),
      m_ener(p.getParameter<double>("MinE")),
      m_PDG(p.getParameter<std::vector<int> >("PartID")),
      m_HcalToken(consumes<edm::PCaloHitContainer>(edm::InputTag("g4SimHits", "HcalHits"))),
      m_BeamToken(consumes<edm::PCaloHitContainer>(edm::InputTag("g4SimHits", "HcalTB06BeamHits"))),
      m_ptb(p.getParameter<edm::ParameterSet>("TestBeamAnalysis")),
      m_timeLimit(m_ptb.getParameter<double>("TimeLimit")),
      m_widthEcal(m_ptb.getParameter<double>("EcalWidth")),
      m_widthHcal(m_ptb.getParameter<double>("HcalWidth")),
      m_factEcal(m_ptb.getParameter<double>("EcalFactor")),
      m_factHcal(m_ptb.getParameter<double>("HcalFactor")),
      m_eMIP(m_ptb.getParameter<double>("MIP")),
      count(0) {
  usesResource("TFileService");
  if (m_ECAL)
    m_EcalToken = consumes<edm::PCaloHitContainer>(edm::InputTag("g4SimHits", "EcalHitsEB"));
  double minEta = p.getParameter<double>("MinEta");
  double maxEta = p.getParameter<double>("MaxEta");
  double minPhi = p.getParameter<double>("MinPhi");
  double maxPhi = p.getParameter<double>("MaxPhi");
  double beamEta = (maxEta + minEta) * 0.5;
  double beamPhi = (maxPhi + minPhi) * 0.5;
  if (beamPhi < 0) {
    beamPhi += twopi;
  }

  m_idxetaEcal = 13;
  m_idxphiEcal = 13;

  m_idxetaHcal = static_cast<int>(beamEta / 0.087) + 1;
  m_idxphiHcal = static_cast<int>(beamPhi / 0.087) + 6;
  if (m_idxphiHcal > 72) {
    m_idxphiHcal -= 73;
  }

  edm::LogInfo("HcalTB06Analysis") << "Beam parameters: E(GeV)= " << m_ener << " pdgID= " << m_PDG[0]
                                   << "\n eta= " << m_eta << "  idx_etaEcal= " << m_idxetaEcal
                                   << "  idx_etaHcal= " << m_idxetaHcal << "  phi= " << m_phi
                                   << "  idx_phiEcal= " << m_idxphiEcal << "  idx_phiHcal= " << m_idxphiHcal
                                   << "\n        EcalFactor= " << m_factEcal << "  EcalWidth= " << m_widthEcal << " GeV"
                                   << "\n        HcalFactor= " << m_factHcal << "  HcalWidth= " << m_widthHcal << " GeV"
                                   << "  MIP=       " << m_eMIP << " GeV"
                                   << "\n        TimeLimit=  " << m_timeLimit << " ns"
                                   << "\n";
  m_histo = std::make_unique<HcalTB06Histo>(m_ptb);
}

void HcalTB06Analysis::beginJob() { edm::LogInfo("HcalTB06Analysis") << " =====> Begin of Run"; }

void HcalTB06Analysis::endJob() {
  edm::LogInfo("HcalTB06Analysis") << " =====> End of Run; Total number of events: " << count;
}

void HcalTB06Analysis::analyze(const edm::Event& evt, const edm::EventSetup&) {
  ++count;

  //Beam Information
  m_histo->fillPrimary(m_ener, m_eta, m_phi);

  std::vector<double> eCalo(6, 0), eTrig(7, 0);

  const std::vector<PCaloHit>* EcalHits = nullptr;
  if (m_ECAL) {
    const edm::Handle<edm::PCaloHitContainer>& Ecal = evt.getHandle(m_EcalToken);
    EcalHits = Ecal.product();
  }
  const edm::Handle<edm::PCaloHitContainer>& Hcal = evt.getHandle(m_HcalToken);
  const std::vector<PCaloHit>* HcalHits = Hcal.product();
  const edm::Handle<edm::PCaloHitContainer>& Beam = evt.getHandle(m_BeamToken);
  const std::vector<PCaloHit>* BeamHits = Beam.product();

  // Total Energy
  double eecals = 0.;
  double ehcals = 0.;

  unsigned int ne = 0;
  unsigned int nh = 0;
  if (m_ECAL) {
    ne = EcalHits->size();
    for (unsigned int i = 0; i < ne; ++i) {
      EBDetId ecalid((*EcalHits)[i].id());
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HcalTBSim") << "EB " << i << " " << ecalid.ieta() << ":" << m_idxetaEcal << "   "
                                    << ecalid.iphi() << ":" << m_idxphiEcal << "   " << (*EcalHits)[i].time() << ":"
                                    << m_timeLimit << "   " << (*EcalHits)[i].energy();
#endif
      // 7x7 crystal selection
      if (std::abs(m_idxetaEcal - ecalid.ieta()) <= 3 && std::abs(m_idxphiEcal - ecalid.iphi()) <= 3 &&
          (*EcalHits)[i].time() < m_timeLimit) {
        eCalo[0] += (*EcalHits)[i].energy();
      }
    }
    if (m_widthEcal > 0.0) {
      eCalo[1] = G4RandGauss::shoot(0.0, m_widthEcal);
    }
    eecals = m_factEcal * (eCalo[0] + eCalo[1]);
  }
  if (HcalHits) {
    nh = HcalHits->size();
    for (unsigned int i = 0; i < nh; ++i) {
      HcalDetId hcalid((*HcalHits)[i].id());
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HcalTBSim") << "HC " << i << " " << hcalid.subdet() << "  " << hcalid.ieta() << ":"
                                    << m_idxetaHcal << "   " << hcalid.iphi() << ":" << m_idxphiHcal << "   "
                                    << (*HcalHits)[i].time() << ":" << m_timeLimit << "   " << (*HcalHits)[i].energy();
#endif
      // 3x3 towers selection
      if (std::abs(m_idxetaHcal - hcalid.ieta()) <= 1 && std::abs(m_idxphiHcal - hcalid.iphi()) <= 1 &&
          (*HcalHits)[i].time() < m_timeLimit) {
        if (hcalid.subdet() != HcalOuter) {
          eCalo[2] += (*HcalHits)[i].energy();
        } else {
          eCalo[4] += (*HcalHits)[i].energy();
        }
      }
    }
    if (m_widthHcal > 0.0) {
      eCalo[3] = G4RandGauss::shoot(0.0, m_widthHcal);
      eCalo[5] = G4RandGauss::shoot(0.0, m_widthHcal);
    }
    ehcals = m_factHcal * eCalo[2] + eCalo[3];
  }
  double etots = eecals + ehcals;

  edm::LogInfo("HcalTBSim") << "HcalTB06Analysis:: Etot(MeV)= " << etots << "   E(Ecal)= " << eecals
                            << "   E(Hcal)= " << ehcals << "  Nhits(ECAL)= " << ne << "  Nhits(HCAL)= " << nh;
  m_histo->fillEdep(etots, eecals, ehcals);

  if (BeamHits) {
    for (unsigned int i = 0; i < BeamHits->size(); ++i) {
      unsigned int id = ((*BeamHits)[i].id());
      int det, lay, ix, iy;
      HcalTestBeamNumbering::unpackIndex(id, det, lay, ix, iy);
      if ((det == 1) && ((*BeamHits)[i].time() < m_timeLimit)) {
        if (lay > 0 && lay <= 4) {
          eTrig[lay - 1] += (*BeamHits)[i].energy();
        } else if (lay == 7 || lay == 8) {
          eTrig[lay - 2] += (*BeamHits)[i].energy();
        } else if (lay >= 11 && lay <= 14) {
          eTrig[4] += (*BeamHits)[i].energy();
        }
      }
    }
  }

  edm::LogInfo("HcalTBSim") << "HcalTB06Analysis:: Trigger Info: " << eTrig[0] << ":" << eTrig[1] << ":" << eTrig[2]
                            << ":" << eTrig[3] << ":" << eTrig[4] << ":" << eTrig[5] << ":" << eTrig[6];

  m_histo->fillTree(eCalo, eTrig);
}

#include "FWCore/PluginManager/interface/ModuleDef.h"

DEFINE_FWK_MODULE(HcalTB06Analysis);

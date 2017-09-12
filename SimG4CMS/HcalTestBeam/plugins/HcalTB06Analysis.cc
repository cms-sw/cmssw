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
#include "SimG4CMS/HcalTestBeam/interface/HcalTB06Analysis.h"
#include "SimG4CMS/HcalTestBeam/interface/HcalTB06Histo.h"
#include "SimG4CMS/HcalTestBeam/interface/HcalTB06BeamSD.h"
#include "SimDataFormats/HcalTestBeam/interface/HcalTestBeamNumbering.h"

// to retreive hits
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

#include "SimG4CMS/Calo/interface/CaloG4Hit.h"
#include "SimG4CMS/Calo/interface/CaloG4HitCollection.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "globals.hh"
#include "Randomize.hh"

// system include files
#include <iostream>
#include <iomanip>

//#define EDM_ML_DEBUG
//
// constructors and destructor
//

HcalTB06Analysis::HcalTB06Analysis(const edm::ParameterSet &p) : count(0) {

  usesResource("TFileService");

  m_ECAL = p.getParameter<bool>("ECAL");
  if(m_ECAL) {
    m_EcalToken = consumes<edm::PCaloHitContainer>(edm::InputTag("g4SimHits","EcalHitsEB"));
  }
  m_HcalToken = consumes<edm::PCaloHitContainer>(edm::InputTag("g4SimHits","HcalHits"));
  m_BeamToken = consumes<edm::PCaloHitContainer>(edm::InputTag("g4SimHits","HcalTB06BeamHits"));
  m_eta = p.getParameter<double>("MinEta");
  m_phi = p.getParameter<double>("MinPhi");
  m_ener= p.getParameter<double>("MinE");
  m_PDG = p.getParameter<std::vector<int> >("PartID");

  double minEta  = p.getParameter<double>("MinEta");
  double maxEta  = p.getParameter<double>("MaxEta");
  double minPhi  = p.getParameter<double>("MinPhi");
  double maxPhi  = p.getParameter<double>("MaxPhi");
  double beamEta = (maxEta+minEta)*0.5;
  double beamPhi = (maxPhi+minPhi)*0.5;
  if (beamPhi < 0) { beamPhi += twopi; }

  m_idxetaEcal = 13;
  m_idxphiEcal = 13;

  m_idxetaHcal   = (int)(beamEta/0.087) + 1;
  m_idxphiHcal   = (int)(beamPhi/0.087) + 6;
  if(m_idxphiHcal > 72) { m_idxphiHcal -= 73; }

  edm::ParameterSet ptb = p.getParameter<edm::ParameterSet>("TestBeamAnalysis");
  m_timeLimit = ptb.getParameter<double>("TimeLimit");
  m_widthEcal = ptb.getParameter<double>("EcalWidth");
  m_widthHcal = ptb.getParameter<double>("HcalWidth");
  m_factEcal  = ptb.getParameter<double>("EcalFactor");
  m_factHcal  = ptb.getParameter<double>("HcalFactor");

  edm::LogInfo("HcalTB06Analysis") 
    << "Beam parameters: E(GeV)= " << m_ener
    << " pdgID= " << m_PDG[0]
    << "\n eta= " << m_eta 
    << "  idx_etaEcal= " << m_idxetaEcal 
    << "  idx_etaHcal= " << m_idxetaHcal 
    << "  phi= " << m_phi 
    << "  idx_phiEcal= " << m_idxphiEcal
    << "  idx_phiHcal= " << m_idxphiHcal
    << "\n        EcalFactor= " << m_factEcal
    << "  EcalWidth= " << m_widthEcal << " GeV" 
    << "\n        HcalFactor= " << m_factHcal
    << "  HcalWidth= " << m_widthHcal << " GeV"
    << "\n        TimeLimit=  " << m_timeLimit << " ns" << "\n";
  m_histo = new HcalTB06Histo(ptb);
} 
   
HcalTB06Analysis::~HcalTB06Analysis() {
  delete m_histo;
}

void HcalTB06Analysis::beginJob() {
  edm::LogInfo("HcalTB06Analysis") <<" =====> Begin of Run";
}

void HcalTB06Analysis::endJob() {
  edm::LogInfo("HcalTB06Analysis") 
    << " =====> End of Run; Total number of events: " << count;
}

void HcalTB06Analysis::analyze(const edm::Event & evt, const edm::EventSetup&)
{
  ++count;

  //Beam Information
  m_histo->fillPrimary(m_ener, m_eta, m_phi);

  edm::Handle<edm::PCaloHitContainer> Ecal;
  edm::Handle<edm::PCaloHitContainer> Hcal;
  edm::Handle<edm::PCaloHitContainer> Beam;
  std::vector<double> eCalo(6,0), eTrig(7,0);

  const std::vector<PCaloHit>* EcalHits = nullptr;
  if(m_ECAL) { 
    evt.getByToken(m_EcalToken, Ecal); 
    EcalHits = Ecal.product();
  }
  evt.getByToken(m_HcalToken, Hcal);
  const std::vector<PCaloHit>* HcalHits = Hcal.product();
  evt.getByToken(m_BeamToken, Beam);
  const std::vector<PCaloHit>* BeamHits = Beam.product();

  // Total Energy
  double eecals = 0.;
  double ehcals = 0.;

  unsigned int ne = 0;
  unsigned int nh = 0;
  if(m_ECAL) {  
    ne = EcalHits->size();
    for (unsigned int i=0; i<ne; ++i) {
      EBDetId ecalid((*EcalHits)[i].id());
#ifdef EDM_ML_DEBUG
      std::cout << "EB " << i << " " << ecalid.ieta() << ":" << m_idxetaEcal 
		<< "   " << ecalid.iphi() << ":" << m_idxphiEcal << "   " 
		<< (*EcalHits)[i].time() << ":" << m_timeLimit << "   " 
		<< (*EcalHits)[i].energy() << std::endl;
#endif
      // 7x7 crystal selection
      if(std::abs(m_idxetaEcal - ecalid.ieta()) <= 3 &&
      	 std::abs(m_idxphiEcal - ecalid.iphi()) <= 3 &&
	 (*EcalHits)[i].time() < m_timeLimit) {
	eCalo[0] += (*EcalHits)[i].energy();
      }
    }
    if(m_widthEcal > 0.0) {
      eCalo[1] = G4RandGauss::shoot(0.0,m_widthEcal);
    }
    eecals = m_factEcal*(eCalo[0]+eCalo[1]);
  }
  if(HcalHits) {
    nh = HcalHits->size();
    for (unsigned int i=0; i<nh; ++i) {
      HcalDetId hcalid((*HcalHits)[i].id());
#ifdef EDM_ML_DEBUG
      std::cout << "HC " << i << " " << hcalid.subdet() << "  " 
		<< hcalid.ieta() << ":" << m_idxetaHcal << "   " 
		<< hcalid.iphi() << ":" << m_idxphiHcal << "   " 
		<< (*HcalHits)[i].time() << ":" << m_timeLimit << "   " 
		<< (*HcalHits)[i].energy() << std::endl;
#endif
      // 3x3 towers selection
      if(std::abs(m_idxetaHcal - hcalid.ieta()) <= 1 &&
      	 std::abs(m_idxphiHcal - hcalid.iphi()) <= 1 &&
	 (*HcalHits)[i].time() < m_timeLimit) {
	if (hcalid.subdet() != HcalOuter) {
	  eCalo[2] += (*HcalHits)[i].energy();
	} else {
	  eCalo[4] += (*HcalHits)[i].energy();
	}
      }
    }
    if(m_widthHcal > 0.0) {
      eCalo[3] = G4RandGauss::shoot(0.0,m_widthHcal);
      eCalo[5] = G4RandGauss::shoot(0.0,m_widthHcal);
    }
    ehcals = m_factHcal*eCalo[2] + eCalo[3];
  }
  double etots = eecals + ehcals;

  edm::LogInfo("HcalTBSim") << "HcalTB06Analysis:: Etot(MeV)= " << etots 
			    << "   E(Ecal)= " << eecals 
			    << "   E(Hcal)= " << ehcals
			    << "  Nhits(ECAL)= " << ne 
			    << "  Nhits(HCAL)= " << nh;
  m_histo->fillEdep(etots, eecals, ehcals); 

  if(BeamHits) {
    for (unsigned int i=0; i<BeamHits->size(); ++i) {
      unsigned int id = ((*BeamHits)[i].id());
      int det, lay, ix, iy;
      HcalTestBeamNumbering::unpackIndex(id,det,lay,ix,iy);
      if ((det == 1) && ((*BeamHits)[i].time() < m_timeLimit)) {
	if (lay > 0 && lay <= 4) {
	  eTrig[lay-1] += (*BeamHits)[i].energy();
	} else if (lay == 7 || lay == 8) {
	  eTrig[lay-2] += (*BeamHits)[i].energy();
	} else if (lay >= 11 && lay <= 14) {
	  eTrig[4]     += (*BeamHits)[i].energy();
	}
      }
    }
  }

  edm::LogInfo("HcalTBSim") << "HcalTB06Analysis:: Trigger Info: " 
			    << eTrig[0] << ":" << eTrig[1] << ":" << eTrig[2]
			    << ":" << eTrig[3] << ":" << eTrig[4] << ":" 
			    << eTrig[5] << ":" << eTrig[6];

  m_histo->fillTree(eCalo,eTrig);
}

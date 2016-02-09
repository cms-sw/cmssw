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

// to retreive hits
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

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

//
// constructors and destructor
//

HcalTB06Analysis::HcalTB06Analysis(const edm::ParameterSet &p)
  : count(0), m_EcalTag(edm::InputTag("g4SimHits","EcalHitsEB")), 
    m_HcalTag(edm::InputTag("g4SimHits","HcalHits"))
{
  m_ECAL = p.getParameter<bool>("ECAL");
  if(m_ECAL) {
    consumes<edm::PCaloHitContainer>(m_EcalTag);
  }
  consumes<edm::PCaloHitContainer>(m_HcalTag);
  m_eta = p.getParameter<double>("MinEta");
  m_phi = p.getParameter<double>("MinPhi");
  m_ener= p.getParameter<double>("MinE");
  m_PDG = p.getParameter<std::vector<int> >("PartID");

  edm::ParameterSet ptb = p.getParameter<edm::ParameterSet>("TestBeamAnalysis");
  m_widthEcal = ptb.getParameter<double>("EcalWidth");
  m_factEcal  = ptb.getParameter<double>("EcalFactor");
  m_factHcal  = ptb.getParameter<double>("HcalFactor");

  edm::LogInfo("HcalTB06Analysis") 
    << "Beam parameters: E(GeV)= " << m_ener
    << " pdgID= " << m_PDG[0]
    << "  eta= " << m_eta
    << "  phi= " << m_phi
    << "\n        EcalFactor= " << m_factEcal
    << "  EcalWidth= " << m_widthEcal << " GeV" 
    << "\n        HcalFactor= " << m_factHcal;
  m_histo = new HcalTB06Histo(ptb);
} 
   
HcalTB06Analysis::~HcalTB06Analysis() 
{
  delete m_histo;
}

void HcalTB06Analysis::beginJob() 
{
  edm::LogInfo("HcalTB06Analysis") <<" =====> Begin of Run";
}

void HcalTB06Analysis::endJob()
{
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

  const std::vector<PCaloHit>* EcalHits = nullptr;
  if(m_ECAL) { 
    evt.getByLabel(m_EcalTag, Ecal); 
    EcalHits = Ecal.product();
  }
  evt.getByLabel(m_HcalTag, Hcal);
  const std::vector<PCaloHit>* HcalHits = Hcal.product();

  // Total Energy
  double eecals = 0.;
  double ehcals = 0.;

  unsigned int ne = 0;
  unsigned int nh = 0;

  if(EcalHits) {  
    ne = EcalHits->size();
    for (unsigned int i=0; i<ne; ++i) {
      eecals += (*EcalHits)[i].energy();
    }
    if(m_widthEcal > 0.0) {
      eecals += G4RandGauss::shoot(0.0,m_widthEcal);
    }
    eecals *= m_factEcal;
  }
  if(HcalHits) {
    nh = HcalHits->size();
    for (unsigned int i=0; i<nh; ++i) {
      ehcals += (*HcalHits)[i].energy();
    }
    ehcals *= m_factHcal;
  }
  double etots = eecals + ehcals;
  LogDebug("HcalTBSim") << "HcalTB06Analysis:: Etot(MeV)= " << etots 
			<< "   E(Ecal)= " << eecals 
			<< "   E(Hcal)= " << ehcals
			<< "  Nhits(ECAL)= " << ne 
			<< "  Nhits(HCAL)= " << nh;
  m_histo->fillEdep(etots, eecals, ehcals);
}

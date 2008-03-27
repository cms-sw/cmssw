#include "Validation/RecoMET/interface/CaloTowerAnalyzer.h"
// author: Bobby Scurlock, University of Florida
// first version 12/18/2006
// modified: Mike Schmitt
// date: 02.28.2007
// note: code rewrite

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "FWCore/PluginManager/interface/ModuleDef.h"
//#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/Handle.h"
//#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/GenMET.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/METReco/interface/GenMETCollection.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"

#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"

//#include "Geometry/Vector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include <vector>
#include <utility>
#include <ostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <cmath>
#include <memory>
#include <TLorentzVector.h>
#include "DQMServices/Core/interface/DQMStore.h"

CaloTowerAnalyzer::CaloTowerAnalyzer(const edm::ParameterSet & iConfig)
{

  // outputFile_          = iConfig.getUntrackedParameter<std::string>("OutputFile");
  geometryFile_        = iConfig.getUntrackedParameter<std::string>("GeometryFile");
  caloTowersLabel_     = iConfig.getParameter<edm::InputTag>("CaloTowersLabel");
  debug_               = iConfig.getParameter<bool>("Debug");
  dumpGeometry_        = iConfig.getParameter<bool>("DumpGeometry");

  //  if (outputFile_.size() > 0)
  //  edm::LogInfo("OutputInfo") << " MET/CaloTower Task histograms will be saved to '" << outputFile_.c_str() << "'";
  //else edm::LogInfo("OutputInfo") << " MET/CaloTower Task histograms will NOT be saved";

}

void CaloTowerAnalyzer::beginJob(const edm::EventSetup& iSetup)
{
  Nevents = 0;
  // get ahold of back-end interface
  dbe_ = edm::Service<DQMStore>().operator->();

  if (dbe_) {
    dbe_->setCurrentFolder("RecoMETV/METTask/CaloTowers/geometry");

    me["hCT_ieta_iphi_etaMap"]      = dbe_->book2D("METTask_CT_ieta_iphi_etaMap","",83,-41,42, 72,1,73);
    me["hCT_ieta_iphi_phiMap"]      = dbe_->book2D("METTask_CT_ieta_iphi_phiMap","",83,-41,42, 72,1,73);
    me["hCT_ieta_detaMap"]          = dbe_->book1D("METTask_CT_ieta_detaMap","", 83, -41, 42);
    me["hCT_ieta_dphiMap"]          = dbe_->book1D("METTask_CT_ieta_dphiMap","", 83, -41, 42);

    // Initialize bins for geometry to -999 because z = 0 is a valid entry
    for (int i=1; i<=83; i++) {
      me["hCT_ieta_detaMap"]->setBinContent(i,-999);
      me["hCT_ieta_dphiMap"]->setBinContent(i,-999);
      for (int j=1; j<=73; j++) {
        me["hCT_ieta_iphi_etaMap"]->setBinContent(i,j,-999);
        me["hCT_ieta_iphi_phiMap"]->setBinContent(i,j,-999);
      }
    }
    TString dirName = "RecoMETV/METTask/CaloTowers/";
    TString label(caloTowersLabel_.label());  
    dirName += label;   
    dbe_->setCurrentFolder((string)dirName); 
    
    //--Store number of events used
    me["hCT_Nevents"]          = dbe_->book1D("METTask_CT_Nevents","",1,0,1);  
    //--Data integrated over all events and stored by CaloTower(ieta,iphi) 
    me["hCT_et_ieta_iphi"]          = dbe_->book2D("METTask_CT_et_ieta_iphi","",83,-41,42, 72,1,73);  
    me["hCT_Minet_ieta_iphi"]          = dbe_->book2D("METTask_CT_Minet_ieta_iphi","",83,-41,42, 72,1,73);  
    me["hCT_Maxet_ieta_iphi"]          = dbe_->book2D("METTask_CT_Maxet_ieta_iphi","",83,-41,42, 72,1,73);  
    for (int i = 1; i<=83; i++)
      for (int j = 1; j<=73; j++)
	{
	  me["hCT_Minet_ieta_iphi"]->setBinContent(i,j,14E3);
	  me["hCT_Maxet_ieta_iphi"]->setBinContent(i,j,-999);
	}

    me["hCT_emEt_ieta_iphi"]        = dbe_->book2D("METTask_CT_emEt_ieta_iphi","",83,-41,42, 72,1,73);  
    me["hCT_hadEt_ieta_iphi"]       = dbe_->book2D("METTask_CT_hadEt_ieta_iphi","",83,-41,42, 72,1,73);  
    me["hCT_energy_ieta_iphi"]      = dbe_->book2D("METTask_CT_energy_ieta_iphi","",83,-41,42, 72,1,73);  
    me["hCT_outerEnergy_ieta_iphi"] = dbe_->book2D("METTask_CT_outerEnergy_ieta_iphi","",83,-41,42, 72,1,73);  
    me["hCT_hadEnergy_ieta_iphi"]   = dbe_->book2D("METTask_CT_hadEnergy_ieta_iphi","",83,-41,42, 72,1,73);  
    me["hCT_emEnergy_ieta_iphi"]    = dbe_->book2D("METTask_CT_emEnergy_ieta_iphi","",83,-41,42, 72,1,73);  
    me["hCT_Occ_ieta_iphi"]         = dbe_->book2D("METTask_CT_Occ_ieta_iphi","",83,-41,42, 72,1,73);  
    //--Data over eta-rings
    // CaloTower values
    me["hCT_etvsieta"]          = dbe_->book2D("METTask_CT_etvsieta","", 83,-41,42, 10001,0,1001);  
    me["hCT_Minetvsieta"]          = dbe_->book2D("METTask_CT_Minetvsieta","", 83,-41,42, 10001,0,1001);  
    me["hCT_Maxetvsieta"]          = dbe_->book2D("METTask_CT_Maxetvsieta","", 83,-41,42, 10001,0,1001);  
    me["hCT_emEtvsieta"]        = dbe_->book2D("METTask_CT_emEtvsieta","",83,-41,42, 10001,0,1001);  
    me["hCT_hadEtvsieta"]       = dbe_->book2D("METTask_CT_hadEtvsieta","",83,-41,42, 10001,0,1001);  
    me["hCT_energyvsieta"]      = dbe_->book2D("METTask_CT_energyvsieta","",83,-41,42, 10001,0,1001);  
    me["hCT_outerEnergyvsieta"] = dbe_->book2D("METTask_CT_outerEnergyvsieta","",83,-41,42, 10001,0,1001);  
    me["hCT_hadEnergyvsieta"]   = dbe_->book2D("METTask_CT_hadEnergyvsieta","",83,-41,42, 10001,0,1001);  
    me["hCT_emEnergyvsieta"]    = dbe_->book2D("METTask_CT_emEnergyvsieta","",83,-41,42, 10001,0,1001);  
    // Integrated over phi
    me["hCT_Occvsieta"]         = dbe_->book2D("METTask_CT_Occvsieta","",83,-41,42, 84,0,84);  
    me["hCT_SETvsieta"]         = dbe_->book2D("METTask_CT_SETvsieta","",83,-41,42, 20001,0,2001);  
    me["hCT_METvsieta"]         = dbe_->book2D("METTask_CT_METvsieta","",83,-41,42, 20001,0,2001);  
    me["hCT_METPhivsieta"]      = dbe_->book2D("METTask_CT_METPhivsieta","",83,-41,42, 80,-4,4);  
    me["hCT_MExvsieta"]         = dbe_->book2D("METTask_CT_MExvsieta","",83,-41,42, 10001,-500,501);  
    me["hCT_MEyvsieta"]         = dbe_->book2D("METTask_CT_MEyvsieta","",83,-41,42, 10001,-500,501);  
  }

  // Inspect Setup for CaloTower Geometry
  FillGeometry(iSetup);

}

void CaloTowerAnalyzer::FillGeometry(const edm::EventSetup& iSetup)
{

  // ==========================================================
  // Retrieve!
  // ==========================================================

  const CaloSubdetectorGeometry* geom;

  try {

    edm::ESHandle<CaloGeometry> pG;
    iSetup.get<IdealGeometryRecord>().get(pG);
    const CaloGeometry cG = *pG;
    geom = cG.getSubdetectorGeometry(DetId::Calo,1);

  } catch (...) {

    edm::LogInfo("OutputInfo") << "Failed to retrieve an Event Setup Handle, Aborting METTask/"
                          << "CaloTowerAnalyzer::FillGeometry!"; return;

  }
    
  // ==========================================================
  // Fill Histograms!
  // ==========================================================

  vector<DetId> ids = geom->getValidDetIds(DetId::Calo,1);
  vector<DetId>::iterator i;

  // Loop Over all CaloTower DetId's
  int ndetid = 0;
  for (i = ids.begin(); i != ids.end(); i++) {

    ndetid++;

    const CaloCellGeometry* cell = geom->getGeometry(*i);
    CaloTowerDetId ctId(i->rawId());
    //GlobalPoint p = cell->getPosition();
      
    int Tower_ieta = ctId.ieta();
    int Tower_iphi = ctId.iphi();
    double Tower_eta = cell->getPosition().eta();
    double Tower_phi = cell->getPosition().phi();
      
    me["hCT_ieta_iphi_etaMap"]->setBinContent(Tower_ieta+42, Tower_iphi, Tower_eta);
    me["hCT_ieta_iphi_phiMap"]->setBinContent(Tower_ieta+42, Tower_iphi, (Tower_phi*180.0/M_PI) );
      
  } // end loop over DetId's

  // Set the Cell Size for each (ieta, iphi) Bin
  double currentLowEdge_eta = 0; //double currentHighEdge_eta = 0;
  
  for (int ieta=1; ieta<=41 ; ieta++) {

    int ieta_ = 42 + ieta;
    double eta = me["hCT_ieta_iphi_etaMap"]->getBinContent(ieta_,3);
    double phi = me["hCT_ieta_iphi_phiMap"]->getBinContent(ieta_,3);
    double deta = 2.0*(eta-currentLowEdge_eta);
    deta = ((float)((int)(1.0E3*deta + 0.5)))/1.0E3;
    double dphi = 2.0*phi;
    if (ieta==40 || ieta==41) dphi = 20;
    if (ieta<=39 && ieta>=21) dphi = 10;
    if (ieta<=20) dphi = 5;
    // BS: This is WRONG...need to correct overlap 
    if (ieta==28) deta = 0.218;
    if (ieta==29) deta= 0.096;      
    currentLowEdge_eta += deta;

    // BS: This is WRONG...need to correct overlap 
    if (ieta==29) currentLowEdge_eta = 2.964;
    me["hCT_ieta_detaMap"]->setBinContent(ieta_,deta); // positive rings
    me["hCT_ieta_dphiMap"]->setBinContent(ieta_,dphi); // positive rings
    me["hCT_ieta_detaMap"]->setBinContent(42-ieta,deta); // negative rings
    me["hCT_ieta_dphiMap"]->setBinContent(42-ieta,dphi); // negative rings

  } // end loop over ieta

}

void CaloTowerAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  //----------GREG & CHRIS' idea---///
   float ETTowerMin = -1; //GeV
   float METRingMin = -2; // GeV

  Nevents++;
  me["hCT_Nevents"]->Fill(0);

  // ==========================================================
  // Retrieve!
  // ==========================================================

  const CaloTowerCollection *towerCollection;

  edm::Handle<reco::CandidateCollection> to;
  iEvent.getByLabel(caloTowersLabel_,to);
  if (!to.isValid()) {
    edm::LogInfo("OutputInfo") << "Failed to retrieve an Event Handle, Aborting METTask/"
			       << "CaloTowerAnalyzer::analyze!"; return;
  } else {
    const CandidateCollection *towers = (CandidateCollection *)to.product();
    reco::CandidateCollection::const_iterator tower = towers->begin();
    edm::Ref<CaloTowerCollection> towerRef = tower->get<CaloTowerRef>();
    towerCollection = towerRef.product();
  }

  // ==========================================================
  // Fill Histograms!
  // ==========================================================

  edm::LogInfo("OutputInfo") << "There are " << towerCollection->size() << " CaloTowers";
  CaloTowerCollection::const_iterator calotower;
  
  int CTmin_iphi = 99, CTmax_iphi = -99;
  int CTmin_ieta = 99, CTmax_ieta = -99;

  TLorentzVector vMET_EtaRing[83];
  int ActiveRing[83];
  int NActiveTowers[83];
  double SET_EtaRing[83];
  double MinEt_EtaRing[83];
  double MaxEt_EtaRing[83];
  for (int i=0;i<83; i++) 
    {
      ActiveRing[i] = 0;
      NActiveTowers[i] = 0;
      SET_EtaRing[i] = 0;
      MinEt_EtaRing[i] = 0;
      MaxEt_EtaRing[i] = 0;
    }

  for (calotower = towerCollection->begin(); calotower != towerCollection->end(); calotower++) {
    
    //math::RhoEtaPhiVector Momentum = calotower->momentum();
    double Tower_ET = calotower->et();
    double Tower_Energy  = calotower->energy();
    double Tower_Eta = calotower->eta();
    double Tower_Phi = calotower->phi();
    double Tower_EMEnergy = calotower->emEnergy();
    double Tower_HadEnergy = calotower->hadEnergy();
    double Tower_OuterEnergy = calotower->outerEnergy();
    double Tower_EMEt = calotower->emEt();
    double Tower_HadEt = calotower->hadEt();
    //int Tower_EMLV1 = calotower->emLvl1();
    //int Tower_HadLV1 = calotower->hadLv11();
    int Tower_ieta = calotower->id().ieta();
    int Tower_iphi = calotower->id().iphi();
    int EtaRing = 41+Tower_ieta;
    ActiveRing[EtaRing] = 1;
    NActiveTowers[EtaRing]++;
    SET_EtaRing[EtaRing]+=Tower_ET;
    TLorentzVector v_;
    v_.SetPtEtaPhiE(Tower_ET, 0, Tower_Phi, Tower_ET);
    if (Tower_ET>ETTowerMin)
      vMET_EtaRing[EtaRing]-=v_;

    // Fill Histograms
    me["hCT_Occ_ieta_iphi"]->Fill(Tower_ieta,Tower_iphi);
    me["hCT_et_ieta_iphi"]->Fill(Tower_ieta,Tower_iphi,Tower_ET);
    me["hCT_emEt_ieta_iphi"]->Fill(Tower_ieta,Tower_iphi,Tower_EMEt);
    me["hCT_hadEt_ieta_iphi"]->Fill(Tower_ieta,Tower_iphi,Tower_HadEt);
    me["hCT_energy_ieta_iphi"]->Fill(Tower_ieta,Tower_iphi,Tower_Energy);
    me["hCT_outerEnergy_ieta_iphi"]->Fill(Tower_ieta,Tower_iphi,Tower_OuterEnergy);
    me["hCT_hadEnergy_ieta_iphi"]->Fill(Tower_ieta,Tower_iphi,Tower_HadEnergy);
    me["hCT_emEnergy_ieta_iphi"]->Fill(Tower_ieta,Tower_iphi,Tower_EMEnergy);

    me["hCT_etvsieta"]->Fill(Tower_ieta, Tower_ET);
    me["hCT_emEtvsieta"]->Fill(Tower_ieta, Tower_EMEt);
    me["hCT_hadEtvsieta"]->Fill(Tower_ieta,Tower_HadEt);
    me["hCT_energyvsieta"]->Fill(Tower_ieta,Tower_Energy);
    me["hCT_outerEnergyvsieta"]->Fill(Tower_ieta,Tower_OuterEnergy);
    me["hCT_hadEnergyvsieta"]->Fill(Tower_ieta ,Tower_HadEnergy);
    me["hCT_emEnergyvsieta"]->Fill(Tower_ieta,Tower_EMEnergy);

    if (Tower_ET > MaxEt_EtaRing[EtaRing])
      MaxEt_EtaRing[EtaRing] = Tower_ET;
    if (Tower_ET < MinEt_EtaRing[EtaRing] && Tower_ET>0)
      MinEt_EtaRing[EtaRing] = Tower_ET;


    if (Tower_ieta < CTmin_ieta) CTmin_ieta = Tower_ieta;
    if (Tower_ieta > CTmax_ieta) CTmax_ieta = Tower_ieta;
    if (Tower_iphi < CTmin_iphi) CTmin_iphi = Tower_iphi;
    if (Tower_iphi > CTmax_iphi) CTmax_iphi = Tower_iphi;
    
    
    
  } // end loop over towers
  
  // Fill eta-ring MET quantities
  for (int iEtaRing=0; iEtaRing<83; iEtaRing++)
    { 
      me["hCT_Minetvsieta"]->Fill(iEtaRing-41, MinEt_EtaRing[iEtaRing]);  
      me["hCT_Maxetvsieta"]->Fill(iEtaRing-41, MaxEt_EtaRing[iEtaRing]);  
      
      if (ActiveRing[iEtaRing])
	{
	  if (vMET_EtaRing[iEtaRing].Pt()>METRingMin)
	    {
	      me["hCT_METPhivsieta"]->Fill(iEtaRing-41, vMET_EtaRing[iEtaRing].Phi());
	      me["hCT_MExvsieta"]->Fill(iEtaRing-41, vMET_EtaRing[iEtaRing].Px());
	      me["hCT_MEyvsieta"]->Fill(iEtaRing-41, vMET_EtaRing[iEtaRing].Py());
	      me["hCT_METvsieta"]->Fill(iEtaRing-41, vMET_EtaRing[iEtaRing].Pt());
	    }
	  me["hCT_SETvsieta"]->Fill(iEtaRing-41, SET_EtaRing[iEtaRing]);
	  me["hCT_Occvsieta"]->Fill(iEtaRing-41, NActiveTowers[iEtaRing]);
	}
    }
  
  edm::LogInfo("OutputInfo") << "CT ieta range: " << CTmin_ieta << " " << CTmax_ieta;
  edm::LogInfo("OutputInfo") << "CT iphi range: " << CTmin_iphi << " " << CTmax_iphi;
  
}


void CaloTowerAnalyzer::DumpGeometry()
{

  ofstream dump(geometryFile_.c_str());

  dump << "Tower Definitions: " << endl << endl;

  dump.width(15); dump << left << "ieta bin";
  dump.width(15); dump << left << "Eta";
  //dump.width(15); dump << left << "Phi";
  dump.width(15); dump << left << "dEta";
  dump.width(15); dump << left << "dPhi" << endl;

  int max_ieta_bin = me["hCT_ieta_iphi_etaMap"]->getNbinsX();
  for (int i = 1; i <= max_ieta_bin; i++) {

    dump.width(15); dump << left << i;
    dump.width(15); dump << left << me["hCT_ieta_iphi_etaMap"]->getBinContent(i,1);
    //dump.width(15); dump << left << me["hCT_ieta_iphi_phiMap"]->getBinContent(i,1);
    dump.width(15); dump << left << me["hCT_ieta_detaMap"]->getBinContent(i);
    dump.width(15); dump << left << me["hCT_ieta_dphiMap"]->getBinContent(i) << endl;

  }

  dump.close();

}

void CaloTowerAnalyzer::endJob()
{
  // Store the DAQ Histograms
  //if (outputFile_.size() > 0 && dbe_)
  //  dbe_->save(outputFile_);

  // Dump Geometry Info to a File
  if (dumpGeometry_); DumpGeometry();

} 

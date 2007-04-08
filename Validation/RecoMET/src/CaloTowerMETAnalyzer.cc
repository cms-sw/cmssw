#include "Validation/RecoMET/interface/CaloTowerMETAnalyzer.h"
// author: Bobby Scurlock, University of Florida
// first version 12/18/2006

#define DEBUG(X) { if (debug_) { cout << X << endl; } }


CaloTowerMETAnalyzer::CaloTowerMETAnalyzer(const edm::ParameterSet& iConfig)
{
  EnergyThreshold = iConfig.getParameter<double>("EnergyThreshold");
  theEvent = iConfig.getParameter<int>("theEvent");
  FirstEvent = iConfig.getParameter<int>("FirstEvent");
  LastEvent = iConfig.getParameter<int>("LastEvent");
}

void CaloTowerMETAnalyzer::endJob() {
  // Normalize the occupancy histogram
  hCT_Occ_ieta_iphi->Scale(100.0/(CurrentEvent+1.0));
  
  //Write out the histogram files.
  m_DataFile->Write();
  m_GeomFile->Write();
} 

void CaloTowerMETAnalyzer::beginJob(const edm::EventSetup& iSetup)
{
  CurrentEvent = -1;
  // Make the output files
  m_GeomFile=new TFile("CaloTower_geometry.root" ,"RECREATE");  
  m_DataFile=new TFile("CaloTowerMETAnalyzer_data.root" ,"RECREATE");  
  // Book the Histograms
  BookHistos();
  // Fill the geometry histograms
  FillGeometry(iSetup);
}

void CaloTowerMETAnalyzer::BookHistos()
{
  // Book Geometry Histograms
  m_GeomFile->cd();
  hCT_ieta_iphi_etaMap = new TH2F("hCT_ieta_iphi_etaMap","",83,-41,42, 73,0,73);
  hCT_ieta_iphi_phiMap = new TH2F("hCT_ieta_iphi_phiMap","",83,-41,42, 73,0,73);
  hCT_ieta_detaMap = new TH1F("hCT_ieta_detaMap","", 83, -41, 42);
  hCT_ieta_dphiMap = new TH1F("hCT_ieta_dphiMap","", 83, -41, 42);
  // Initialize bins for geometry to -999 because z = 0 is a valid entry 
  for (int i=1; i<=83; i++)
    {
      hCT_ieta_detaMap->SetBinContent(i, -999);
      hCT_ieta_dphiMap->SetBinContent(i, -999);
      for (int j=1; j<=73; j++)
	{
	  hCT_ieta_iphi_etaMap->SetBinContent(i,j,-999);
	  hCT_ieta_iphi_phiMap->SetBinContent(i,j,-999);
	}
    }
  
  // Book Data Histograms
  m_DataFile->cd();
  hCT_et_ieta_iphi = new TH2F("hCT_et_ieta_iphi","",83,-41,42, 73,0,73);  
  hCT_emEt_ieta_iphi = new TH2F("hCT_emEt_ieta_iphi","",83,-41,42, 73,0,73);  
  hCT_hadEt_ieta_iphi = new TH2F("hCT_hadEt_ieta_iphi","",83,-41,42, 73,0,73);  
  hCT_energy_ieta_iphi = new TH2F("hCT_energy_ieta_iphi","",83,-41,42, 73,0,73);  
  hCT_outerEnergy_ieta_iphi = new TH2F("hCT_outerEnergy_ieta_iphi","",83,-41,42, 73,0,73);  
  hCT_hadEnergy_ieta_iphi = new TH2F("hCT_hadEnergy_ieta_iphi","",83,-41,42, 73,0,73);  
  hCT_emEnergy_ieta_iphi = new TH2F("hCT_emEnergy_ieta_iphi","",83,-41,42, 73,0,73);  
  hCT_Occ_ieta_iphi = new TH2F("hCT_Occ_ieta_iphi","",83,-41,42, 73,0,73);  
}

void CaloTowerMETAnalyzer::FillGeometry(const edm::EventSetup& iSetup)
{
  // Fill geometry histograms
  using namespace edm;
  //int b=0;
  edm::ESHandle<CaloGeometry> pG;
  iSetup.get<IdealGeometryRecord>().get(pG);
  const CaloGeometry cG = *pG;
  const CaloSubdetectorGeometry* geom=cG.getSubdetectorGeometry(DetId::Calo, 1);
  int n=0;
  std::vector<DetId> ids=geom->getValidDetIds(DetId::Calo,1);
  for (std::vector<DetId>::iterator i=ids.begin(); i!=ids.end(); i++) 
    {
      n++;
      const CaloCellGeometry* cell=geom->getGeometry(*i);
      CaloTowerDetId ctId(i->rawId());
      //GlobalPoint p = cell->getPosition();
      
      int Tower_ieta = ctId.ieta();
      int Tower_iphi = ctId.iphi();
      double Tower_eta = cell->getPosition().eta();
      double Tower_phi = cell->getPosition().phi();
      
      hCT_ieta_iphi_etaMap->SetBinContent(Tower_ieta+42, Tower_iphi+1, Tower_eta);
      hCT_ieta_iphi_phiMap->SetBinContent(Tower_ieta+42, Tower_iphi+1, (Tower_phi*180.0/M_PI) );
      
      DEBUG( "Tower " << n );
      DEBUG( " ieta, iphi = " << Tower_ieta << ", " << Tower_iphi);
      DEBUG( "  eta,  phi = " << cell->getPosition().eta() << ", " << cell->getPosition().phi());
      DEBUG( " " );   
    } // end loop over DetId's

  
  //-------Set the cell size for each (ieta, iphi) bin-------//
  double currentLowEdge_eta = 0;
  //double currentHighEdge_eta = 0;
  for (int ieta=1; ieta<=41 ; ieta++)
    {
      int ieta_ = 42 + ieta;
      double eta = hCT_ieta_iphi_etaMap->GetBinContent(ieta_, 2);
      double phi = hCT_ieta_iphi_phiMap->GetBinContent(ieta_, 2);
      double deta = 2.0*(eta-currentLowEdge_eta);
      deta = ((float)((int)(1.0E3*deta + 0.5)))/1.0E3;
      double dphi = 2.0*phi;
      // BS: This is WRONG...need to correct overlap 
      if (ieta==28) deta = 0.218;
      if (ieta==29) deta= 0.096;      
      currentLowEdge_eta += deta;
      // BS: This is WRONG...need to correct overlap 
      if (ieta==29) currentLowEdge_eta = 2.964;
      hCT_ieta_detaMap->SetBinContent(ieta_, deta); // positive rings
      hCT_ieta_dphiMap->SetBinContent(ieta_, dphi); // positive rings
      hCT_ieta_detaMap->SetBinContent(42-ieta, deta); // negative rings
      hCT_ieta_dphiMap->SetBinContent(42-ieta, dphi); // negative rings
    } // end loop over ieta
}



void CaloTowerMETAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  CurrentEvent++;
  DEBUG( "Event: " << CurrentEvent);  
  WriteCaloTowers(iEvent, iSetup);
}

void CaloTowerMETAnalyzer::WriteCaloTowers(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  edm::Handle<reco::CandidateCollection> to;
  iEvent.getByLabel( "caloTowers", to );
  const CandidateCollection *towers = (CandidateCollection *)to.product();
  reco::CandidateCollection::const_iterator tower = towers->begin();
  edm::Ref<CaloTowerCollection> towerRef = tower->get<CaloTowerRef>();
  const CaloTowerCollection *towerCollection = towerRef.product();
  CaloTowerCollection::const_iterator calotower = towerCollection->begin();
  
  DEBUG( "Event has " << towerCollection->size() << " CaloTowers" );
  Int_t tower_ = 0;
  for( ; calotower != towerCollection->end(); calotower++ ) 
    {
      //math::RhoEtaPhiVector Momentum = calotower->momentum();
      double Tower_ET = calotower->et();
      double Tower_Energy  = calotower->energy();
      //double Tower_Eta = calotower->eta();
      //double Tower_Phi = calotower->phi();
      double Tower_EMEnergy = calotower->emEnergy();
      double Tower_HadEnergy = calotower->hadEnergy();
      double Tower_OuterEnergy = calotower->outerEnergy();
      double Tower_EMEt = calotower->emEt();
      double Tower_HadEt = calotower->hadEt();
      //int Tower_EMLV1 = calotower->emLvl1();
      //int Tower_HadLV1 = calotower->hadLv11();
      int Tower_ieta = calotower->id().ieta();
      int Tower_iphi = calotower->id().iphi();
      
      //int EtaRing = 41+Tower_ieta;
      if (CurrentEvent == theEvent)
	{
	  hCT_et_ieta_iphi->Fill(Tower_ieta, Tower_iphi, Tower_ET );
	  hCT_emEt_ieta_iphi->Fill(Tower_ieta, Tower_iphi, Tower_EMEt );      
	  hCT_hadEt_ieta_iphi->Fill(Tower_ieta, Tower_iphi, Tower_HadEt );
	  hCT_energy_ieta_iphi->Fill(Tower_ieta, Tower_iphi, Tower_Energy );
	  hCT_outerEnergy_ieta_iphi->Fill(Tower_ieta, Tower_iphi, Tower_OuterEnergy );
	  hCT_hadEnergy_ieta_iphi->Fill(Tower_ieta, Tower_iphi, Tower_HadEnergy );
	  hCT_emEnergy_ieta_iphi->Fill(Tower_ieta, Tower_iphi, Tower_EMEnergy );
	  DEBUG("   Tower #" << tower_++ << " ieta = " << Tower_ieta << ", iphi = " << Tower_iphi << ", Energy = " << Tower_Energy); 
	}

      if ( Tower_Energy>EnergyThreshold && CurrentEvent <= LastEvent && CurrentEvent >= FirstEvent ) 
	hCT_Occ_ieta_iphi->Fill(Tower_ieta, Tower_iphi );
    }
}


void CaloTowerMETAnalyzer::DumpGeometry()
{
  cout << "Tower Definitions: " << endl;
  for (int i=1; i<=hCT_ieta_iphi_etaMap->GetNbinsX(); i++)
    {
      cout << " ieta Bin " << i << endl;
      cout << "     dPhi   = " << hCT_ieta_dphiMap->GetBinContent(i, 1) << endl;
      cout << "     dEta   = " << hCT_ieta_detaMap->GetBinContent(i, 1)  << endl;
      cout << "      Eta   = " << hCT_ieta_iphi_etaMap->GetBinContent(i, 1)<< endl;
      cout << endl;
    }
  cout << endl;
}


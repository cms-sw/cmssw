#include "Validation/RecoMET/interface/HCALRecHitAnalyzer.h"
// author: Bobby Scurlock, University of Florida
// first version 12/7/2006

#define DEBUG(X) { if (debug_) { cout << X << endl; } }


HCALRecHitAnalyzer::HCALRecHitAnalyzer(const edm::ParameterSet& iConfig)
{
  EnergyThreshold = iConfig.getParameter<double>("EnergyThreshold");
  theEvent = iConfig.getParameter<int>("theEvent");
  FirstEvent = iConfig.getParameter<int>("FirstEvent");
  LastEvent = iConfig.getParameter<int>("LastEvent");
}

void HCALRecHitAnalyzer::endJob() {
  // Normalize the occupancy histogram
  hHCAL_L1_Occ_ieta_iphi->Scale(100.0/(CurrentEvent+1.0));
  hHCAL_L2_Occ_ieta_iphi->Scale(100.0/(CurrentEvent+1.0));
  hHCAL_L3_Occ_ieta_iphi->Scale(100.0/(CurrentEvent+1.0));
  hHCAL_L4_Occ_ieta_iphi->Scale(100.0/(CurrentEvent+1.0));
  
  //Write out the histogram files.
  m_DataFile->Write();
  m_GeomFile->Write();
} 

void HCALRecHitAnalyzer::beginJob(const edm::EventSetup& iSetup){
  CurrentEvent = -1;
  // Make the output files
  m_GeomFile=new TFile("CaloTower_geometry.root" ,"RECREATE");  
  m_DataFile=new TFile("HCALRecHitAnalyzer_data.root" ,"RECREATE");  
  // Book the Histograms
  BookHistos();
  // Fill the geometry histograms
  FillGeometry(iSetup);
}

void HCALRecHitAnalyzer::BookHistos()
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
  hHCAL_L1_energy_ieta_iphi = new TH2F("hHCAL_L1_energy_ieta_iphi","",83,-41,42, 73,0,73);  
  hHCAL_L2_energy_ieta_iphi = new TH2F("hHCAL_L2_energy_ieta_iphi","",83,-41,42, 73,0,73);  
  hHCAL_L3_energy_ieta_iphi = new TH2F("hHCAL_L3_energy_ieta_iphi","",83,-41,42, 73,0,73);  
  hHCAL_L4_energy_ieta_iphi = new TH2F("hHCAL_L4_energy_ieta_iphi","",83,-41,42, 73,0,73);  
  hHCAL_L1_Occ_ieta_iphi = new TH2F("hHCAL_L1_Occ_ieta_iphi","",83,-41,42, 73,0,73);  
  hHCAL_L2_Occ_ieta_iphi = new TH2F("hHCAL_L2_Occ_ieta_iphi","",83,-41,42, 73,0,73);  
  hHCAL_L3_Occ_ieta_iphi = new TH2F("hHCAL_L3_Occ_ieta_iphi","",83,-41,42, 73,0,73);  
  hHCAL_L4_Occ_ieta_iphi = new TH2F("hHCAL_L4_Occ_ieta_iphi","",83,-41,42, 73,0,73);  

}

void HCALRecHitAnalyzer::FillGeometry(const edm::EventSetup& iSetup)
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



void HCALRecHitAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  CurrentEvent++;
  DEBUG( "Event: " << CurrentEvent);  
  WriteHCALRecHits(iEvent, iSetup);
}

void HCALRecHitAnalyzer::WriteHCALRecHits(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  edm::Handle<HBHERecHitCollection> HBHERecHits;
  edm::Handle<HORecHitCollection> HORecHits;
  edm::Handle<HFRecHitCollection> HFRecHits;
  iEvent.getByLabel( "hbhereco", HBHERecHits );
  iEvent.getByLabel( "horeco", HORecHits );
  iEvent.getByLabel( "hfreco", HFRecHits );

  edm::Handle<reco::CandidateCollection> to;
  iEvent.getByLabel( "caloTowers", to );
  const CandidateCollection *towers = (CandidateCollection *)to.product();
  reco::CandidateCollection::const_iterator tower = towers->begin();
  edm::Ref<CaloTowerCollection> towerRef = tower->get<CaloTowerRef>();
  const CaloTowerCollection *towerCollection = towerRef.product();
  CaloTowerCollection::const_iterator calotower = towerCollection->begin();
  
  edm::ESHandle<CaloGeometry> pG;
  iSetup.get<IdealGeometryRecord>().get(pG);
  const CaloGeometry cG = *pG;
  //const CaloSubdetectorGeometry* EBgeom=cG.getSubdetectorGeometry(DetId::Ecal,1);
  //const CaloSubdetectorGeometry* EEgeom=cG.getSubdetectorGeometry(DetId::Ecal,2);
  // Loop over towers and get each tower's RecHit constituents
  for( ; calotower != towerCollection->end(); calotower++ ) 
    {
      size_t numRecHits = calotower->constituentsSize();
      for(size_t j = 0; j <numRecHits ; j++) {
	DetId RecHitDetID=calotower->constituent(j);
	DetId::Detector DetNum=RecHitDetID.det();
	
	if( DetNum == DetId::Hcal )
	  {
	    DEBUG( " RecHit " << j << ": Detector = " << DetNum << ": Hcal " );
	    HcalDetId HcalID = RecHitDetID;
	    HcalSubdetector HcalNum = HcalID.subdet();
	    Int_t depth = HcalID.depth();
	    Int_t ieta = HcalID.ieta();
	    Int_t iphi = HcalID.iphi();
	    
	    if(  HcalNum == HcalBarrel ) // depths 1,2
	      {
		HBHERecHitCollection::const_iterator theRecHit=HBHERecHits->find(HcalID);
		Float_t Energy = theRecHit->energy();
		DEBUG("         RecHit: " << j << ": HB, ieta=" << HcalID.ieta() << ", iphi=" << HcalID.iphi()<<      
		      ", depth=" << depth << ", energy=" << theRecHit->energy() << ", time=" <<\
		      theRecHit->time());              
		if (depth==1)
		  {
		    if (CurrentEvent == theEvent)
		      hHCAL_L1_energy_ieta_iphi->Fill(ieta, iphi, Energy);
		    if (Energy>EnergyThreshold && CurrentEvent <= LastEvent && CurrentEvent >= FirstEvent) 
		      hHCAL_L1_Occ_ieta_iphi->Fill(ieta, iphi);
		  }
		if (depth==2)
		  {
		    if (CurrentEvent == theEvent)
		      hHCAL_L2_energy_ieta_iphi->Fill(ieta, iphi, Energy);
		    if (Energy>EnergyThreshold && CurrentEvent <= LastEvent && CurrentEvent >= FirstEvent) 
		      hHCAL_L2_Occ_ieta_iphi->Fill(ieta, iphi);
		  }
		
	    }
	    else if(  HcalNum == HcalEndcap  ) // depths 1,2,3
	      {
		HBHERecHitCollection::const_iterator theRecHit=HBHERecHits->find(HcalID);	    
		Float_t Energy = theRecHit->energy();
		DEBUG( "         RecHit: " << j << ": HE, ieta=" << HcalID.ieta() << ", iphi=" << HcalID.iphi()<<      
		  ", depth=" << depth << ", energy=" << theRecHit->energy() << ", time=" <<\
		    theRecHit->time());     
		  if (depth==1)
		    {
		      if (CurrentEvent == theEvent)
			hHCAL_L1_energy_ieta_iphi->Fill(ieta, iphi, Energy);
		      if (Energy>EnergyThreshold && CurrentEvent <= LastEvent && CurrentEvent >= FirstEvent) 
			hHCAL_L1_Occ_ieta_iphi->Fill(ieta, iphi);
		    }
		  if (depth==2)
		    {
		      if (CurrentEvent == theEvent)
			hHCAL_L2_energy_ieta_iphi->Fill(ieta, iphi, Energy);
		      if (Energy>EnergyThreshold && CurrentEvent <= LastEvent && CurrentEvent >= FirstEvent) 
			hHCAL_L2_Occ_ieta_iphi->Fill(ieta, iphi);
		    }
		  if (depth==3)
		    {
		      if (CurrentEvent == theEvent)
			hHCAL_L3_energy_ieta_iphi->Fill(ieta, iphi, Energy);
		      if (Energy>EnergyThreshold && CurrentEvent <= LastEvent && CurrentEvent >= FirstEvent) 
			hHCAL_L3_Occ_ieta_iphi->Fill(ieta, iphi);
		    }
	      }
	    else if(  HcalNum == HcalOuter  ) // depth 4 
	      {
		HORecHitCollection::const_iterator theRecHit=HORecHits->find(HcalID);	    
		Float_t Energy = theRecHit->energy();
		DEBUG("         RecHit: " << j << ": HO, ieta=" << HcalID.ieta() << ", iphi=" << HcalID.iphi()<<      
		      ", depth=" << depth << ", energy=" << theRecHit->energy() << ", time=" <<\
		      theRecHit->time() );
		  if (depth==4)
		    {
		      if (CurrentEvent == theEvent)
			hHCAL_L4_energy_ieta_iphi->Fill(ieta, iphi, Energy);
		      if (Energy>EnergyThreshold && CurrentEvent <= LastEvent && CurrentEvent >= FirstEvent) 
			hHCAL_L4_Occ_ieta_iphi->Fill(ieta, iphi);
		    }	  
	      }	     
	    else if(  HcalNum == HcalForward  ) // depths 1,2
	      {
		HFRecHitCollection::const_iterator theRecHit=HFRecHits->find(HcalID);	
		Float_t Energy = theRecHit->energy();
		DEBUG( "         RecHit: " << j << ": HF, ieta=" << HcalID.ieta() << ", iphi=" << HcalID.iphi()<<      
		  ", depth=" << depth << ", energy=" << theRecHit->energy() << ", time=" <<\
		    theRecHit->time());     
		if (depth==1)
		  {
		    if (CurrentEvent == theEvent)
		      hHCAL_L1_energy_ieta_iphi->Fill(ieta, iphi, Energy);
		    if (Energy>EnergyThreshold && CurrentEvent <= LastEvent && CurrentEvent >= FirstEvent) 
		      hHCAL_L1_Occ_ieta_iphi->Fill(ieta, iphi);
		  }
		if (depth==2)
		  {
		    if (CurrentEvent == theEvent)
		      hHCAL_L2_energy_ieta_iphi->Fill(ieta, iphi, Energy);
		    if (Energy>EnergyThreshold && CurrentEvent <= LastEvent && CurrentEvent >= FirstEvent) 
		      hHCAL_L2_Occ_ieta_iphi->Fill(ieta, iphi);
		  }
	      }	                 	      
	  }
	
      }
    }
}


void HCALRecHitAnalyzer::DumpGeometry()
{
  cout << "Tower Definitions: " << endl;
  for (int i=1; i<=hCT_ieta_iphi_etaMap->GetNbinsX(); i++)
    {
      cout << "ieta Bin " << i << endl;
      cout <<  "     dPhi   = " << hCT_ieta_dphiMap->GetBinContent(i, 1) << endl;
      cout <<  "     dEta   = " << hCT_ieta_detaMap->GetBinContent(i, 1)  << endl;
      cout <<  "      Eta   = " << hCT_ieta_iphi_etaMap->GetBinContent(i, 1)<< endl;
      cout << endl;
    }
  cout << endl;
}


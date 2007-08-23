#include "Validation/RecoMET/interface/ECALRecHitAnalyzer.h"
// author: Bobby Scurlock, University of Florida
// first version 11/20/2006

#define DEBUG(X) { if (debug_) { cout << X << endl; } }

ECALRecHitAnalyzer::ECALRecHitAnalyzer(const edm::ParameterSet& iConfig)
{
  debug_             = iConfig.getParameter<bool>("Debug");

 
}

void ECALRecHitAnalyzer::endJob() {
  //Write out the histogram files.
  m_DataFile->Write();
  m_GeomFile->Write();
} 

void ECALRecHitAnalyzer::beginJob(const edm::EventSetup& iSetup){
  CurrentEvent = -1;
  // Make the output files
  m_GeomFile=new TFile("ECAL_geometry.root" ,"RECREATE");  
  m_DataFile=new TFile("ECALRecHitAnalyzer_data.root" ,"RECREATE");  
  // Book the Histograms
  BookHistos();
  // Fill the geometry histograms
  FillGeometry(iSetup);
}

void ECALRecHitAnalyzer::BookHistos()
{
  // Book Geometry Histograms
  m_GeomFile->cd();
  // ECAL barrel
  hEB_ieta_iphi_etaMap = new TH2F("hEB_ieta_iphi_etaMap","", 171, -85, 86, 360, 1, 361);
  hEB_ieta_iphi_phiMap = new TH2F("hEB_ieta_iphi_phiMap","", 171, -85, 86, 360, 1, 361);
  hEB_ieta_detaMap = new TH1F("hEB_ieta_detaMap","", 171, -85, 86);
  hEB_ieta_dphiMap = new TH1F("hEB_ieta_dphiMap","", 171, -85, 86);
  // ECAL +endcap
  hEEpZ_ix_iy_xMap = new TH2F("hEEpZ_ix_iy_xMap","", 100,1,101, 100,1,101);
  hEEpZ_ix_iy_yMap = new TH2F("hEEpZ_ix_iy_yMap","", 100,1,101, 100,1,101);
  hEEpZ_ix_iy_dxMap = new TH2F("hEEpZ_ix_iy_dxMap","", 100,1,101, 100,1,101);  
  hEEpZ_ix_iy_dyMap = new TH2F("hEEpZ_ix_iy_dyMap","", 100,1,101, 100,1,101);
  // ECAL -endcap
  hEEmZ_ix_iy_xMap = new TH2F("hEEmZ_ix_iy_xMap","", 100,1,101, 100,1,101);
  hEEmZ_ix_iy_yMap = new TH2F("hEEmZ_ix_iy_yMap","", 100,1,101, 100,1,101);
  hEEmZ_ix_iy_dxMap = new TH2F("hEEmZ_ix_iy_dxMap","", 100,1,101, 100,1,101);  
  hEEmZ_ix_iy_dyMap = new TH2F("hEEmZ_ix_iy_dyMap","", 100,1,101, 100,1,101);

  // Initialize bins for geometry to -999 because z = 0 is a valid entry 
  for (int i=1; i<=100; i++)
    for (int j=1; j<=100; j++)
      {
	hEEpZ_ix_iy_xMap->SetBinContent(i,j,-999);
	hEEpZ_ix_iy_yMap->SetBinContent(i,j,-999);
	hEEpZ_ix_iy_dxMap->SetBinContent(i,j,-999);
	hEEpZ_ix_iy_dyMap->SetBinContent(i,j,-999);

	hEEmZ_ix_iy_xMap->SetBinContent(i,j,-999);
	hEEmZ_ix_iy_yMap->SetBinContent(i,j,-999);
	hEEmZ_ix_iy_dxMap->SetBinContent(i,j,-999);
	hEEmZ_ix_iy_dyMap->SetBinContent(i,j,-999);
      }

  for (int i=1; i<=171; i++)
    {
      hEB_ieta_detaMap->SetBinContent(i,-999);
      hEB_ieta_dphiMap->SetBinContent(i,-999);
      for (int j=1; j<=360; j++)
	{
	  hEB_ieta_iphi_etaMap->SetBinContent(i,j,-999);
	  hEB_ieta_iphi_phiMap->SetBinContent(i,j,-999);
	}
    }

  // Book Data Histograms
  m_DataFile->cd();
  // Energy Histograms by logical index
  hEEpZ_energy_ix_iy = new TH2F("hEEpZ_energy_ix_iy","", 100,1,101, 100,1,101);
  hEEmZ_energy_ix_iy = new TH2F("hEEmZ_energy_ix_iy","", 100,1,101, 100,1,101);
  hEB_energy_ieta_iphi = new TH2F("hEB_energy_ieta_iphi","", 171, -85, 86, 360, 1, 361);   
  // Occupancy Histograms by logical index
  hEEpZ_Occ_ix_iy = new TH2F("hEEpZ_Occ_ix_iy","", 100,1,101, 100,1,101);  
  hEEmZ_Occ_ix_iy = new TH2F("hEEmZ_Occ_ix_iy","", 100,1,101, 100,1,101);  
  hEB_Occ_ieta_iphi = new TH2F("hEB_Occ_ieta_iphi","",171, -85, 86, 360, 1, 361);   
 
}

void ECALRecHitAnalyzer::FillGeometry(const edm::EventSetup& iSetup)
{
  // Fill geometry histograms
  using namespace edm;
  //int b=0;
  edm::ESHandle<CaloGeometry> pG;
  iSetup.get<IdealGeometryRecord>().get(pG);
  const CaloGeometry cG = *pG;
  
  //----Fill Ecal Barrel----//
  const CaloSubdetectorGeometry* EBgeom=cG.getSubdetectorGeometry(DetId::Ecal,1);
  int n=0;
  std::vector<DetId> EBids=EBgeom->getValidDetIds(DetId::Ecal, 1);
  for (std::vector<DetId>::iterator i=EBids.begin(); i!=EBids.end(); i++) {
    n++;
    const CaloCellGeometry* cell=EBgeom->getGeometry(*i);
    //GlobalPoint p = cell->getPosition();
    
    EBDetId EcalID(i->rawId());
    
    int Crystal_ieta = EcalID.ieta();
    int Crystal_iphi = EcalID.iphi();
    double Crystal_eta = cell->getPosition().eta();
    double Crystal_phi = cell->getPosition().phi();
    hEB_ieta_iphi_etaMap->SetBinContent(Crystal_ieta+86, Crystal_iphi, Crystal_eta);
    hEB_ieta_iphi_phiMap->SetBinContent(Crystal_ieta+86, Crystal_iphi, (Crystal_phi*180/M_PI) );
    
    DEBUG( " Crystal " << n );
    DEBUG( "  ieta, iphi = " << Crystal_ieta << ", " << Crystal_iphi);
    DEBUG( "   eta,  phi = " << cell->getPosition().eta() << ", " << cell->getPosition().phi());
    DEBUG( " " );
    
  }
  //----Fill Ecal Endcap----------//
  const CaloSubdetectorGeometry* EEgeom=cG.getSubdetectorGeometry(DetId::Ecal,2);
  n=0;
  std::vector<DetId> EEids=EEgeom->getValidDetIds(DetId::Ecal, 2);
  for (std::vector<DetId>::iterator i=EEids.begin(); i!=EEids.end(); i++) {
    n++;
    const CaloCellGeometry* cell=EEgeom->getGeometry(*i);
    //GlobalPoint p = cell->getPosition();
    EEDetId EcalID(i->rawId());
    int Crystal_zside = EcalID.zside();
    int Crystal_ix = EcalID.ix();
    int Crystal_iy = EcalID.iy();
    //double Crystal_eta = cell->getPosition().eta();
    //double Crystal_phi = cell->getPosition().phi();
    double Crystal_x = cell->getPosition().x();
    double Crystal_y = cell->getPosition().y();
    // ECAL -endcap
    if (Crystal_zside == -1)
      {
	hEEmZ_ix_iy_xMap->SetBinContent(Crystal_ix, Crystal_iy, Crystal_x);
	hEEmZ_ix_iy_yMap->SetBinContent(Crystal_ix, Crystal_iy, Crystal_y);
      }
    // ECAL +endcap
    if (Crystal_zside == 1)
      {
	hEEpZ_ix_iy_xMap->SetBinContent(Crystal_ix, Crystal_iy, Crystal_x);
	hEEpZ_ix_iy_yMap->SetBinContent(Crystal_ix, Crystal_iy, Crystal_y);
      }

      DEBUG( " Crystal " << n );
      DEBUG( "  side = " << Crystal_zside );
      DEBUG("   ix, iy = " << Crystal_ix << ", " << Crystal_iy);
      DEBUG("    x,  y = " << Crystal_x << ", " << Crystal_y);;
      DEBUG( " " );

  }
 
  //-------Set the cell size for each (ieta, iphi) bin-------//
  double currentLowEdge_eta = 0;
  //double currentHighEdge_eta = 0;
  for (int ieta=1; ieta<=85 ; ieta++)
    {
      int ieta_ = 86 + ieta;
      
      double eta = hEB_ieta_iphi_etaMap->GetBinContent(ieta_, 1);
      double etam1 = -999;
      
      if (ieta==1) 
	etam1 = hEB_ieta_iphi_etaMap->GetBinContent(85, 1);
      else 
	etam1 = hEB_ieta_iphi_etaMap->GetBinContent(ieta_ - 1, 1);

      //double phi = hEB_ieta_iphi_phiMap->GetBinContent(ieta_, 1);
      double deta = fabs( eta - etam1 );
      double dphi = fabs( hEB_ieta_iphi_phiMap->GetBinContent(ieta_, 1) - hEB_ieta_iphi_phiMap->GetBinContent(ieta_, 2) );
          
      currentLowEdge_eta += deta;
      hEB_ieta_detaMap->SetBinContent(ieta_, deta); // positive rings
      hEB_ieta_dphiMap->SetBinContent(ieta_, dphi); // positive rings
      hEB_ieta_detaMap->SetBinContent(86-ieta, deta); // negative rings
      hEB_ieta_dphiMap->SetBinContent(86-ieta, dphi); // negative rings
    }
}



void ECALRecHitAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  CurrentEvent++;
  DEBUG( "Event: " << CurrentEvent);
  WriteECALRecHits( iEvent, iSetup );
}

void ECALRecHitAnalyzer::WriteECALRecHits(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  edm::Handle<EBRecHitCollection> EBRecHits;
  edm::Handle<EERecHitCollection> EERecHits;
  iEvent.getByLabel( "ecalRecHit", "EcalRecHitsEB", EBRecHits );
  iEvent.getByLabel( "ecalRecHit", "EcalRecHitsEE", EERecHits );
  
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
  const CaloSubdetectorGeometry* EBgeom=cG.getSubdetectorGeometry(DetId::Ecal,1);
  const CaloSubdetectorGeometry* EEgeom=cG.getSubdetectorGeometry(DetId::Ecal,2);
  // Loop over towers and get each tower's RecHit constituents
  for( ; calotower != towerCollection->end(); calotower++ ) 
    {
      size_t numRecHits = calotower->constituentsSize();
      for(size_t j = 0; j <numRecHits ; j++) {
	DetId RecHitDetID=calotower->constituent(j);
	DetId::Detector DetNum=RecHitDetID.det();
	// Check if RecHit is in ECAL
	if( DetNum == DetId::Ecal )
	  {
	    int EcalNum =  RecHitDetID.subdetId();
	    // Check if ECAL RecHit is in Barrel
	    if( EcalNum == 1 )
	      {
		EBDetId EcalID = RecHitDetID;
		const CaloCellGeometry* cell=EBgeom->getGeometry(RecHitDetID);
		GlobalPoint p = cell->getPosition();
		EBRecHitCollection::const_iterator theRecHit=EBRecHits->find(EcalID);	   
		float Energy = theRecHit->energy();
		float ieta = EcalID.ieta();
		float iphi = EcalID.iphi();
		hEB_energy_ieta_iphi->Fill(ieta, iphi, Energy);
		hEB_Occ_ieta_iphi->Fill(ieta, iphi);
		
		DEBUG(" ECAL Barrel: ");
		DEBUG("         RecHit " << j << ": EB, ieta=" << EcalID.ieta() <<  ", iphi=" << EcalID.iphi() <<  ", SM=" << EcalID.ism() << ", energy=" << theRecHit->energy());
		DEBUG("                                  eta=" << p.eta()       <<  ",  phi=" << p.phi() );
		
	      } // end if
	    
	    // Check if ECAL RecHit is in Endcap
	    else if(  EcalNum == 2 )
	      {
		EEDetId EcalID = RecHitDetID;
		const CaloCellGeometry* cell=EEgeom->getGeometry(RecHitDetID);
		GlobalPoint p = cell->getPosition();
		EERecHitCollection::const_iterator theRecHit=EERecHits->find(EcalID);
		float Energy = theRecHit->energy();
		float ix = EcalID.ix();
		float iy = EcalID.iy();
		int Crystal_zside = EcalID.zside();
		
		 if (Crystal_zside == -1)
		   {
		     hEEmZ_energy_ix_iy->Fill(ix, iy, Energy);
		     hEEmZ_Occ_ix_iy->Fill(ix, iy);
		   }
		 if (Crystal_zside == 1)
		   {
		     hEEpZ_energy_ix_iy->Fill(ix, iy, Energy);
		     hEEpZ_Occ_ix_iy->Fill(ix, iy);
		   }
		 
		   DEBUG(" ECAL Endcap: " );	    
		   DEBUG("         RecHit " << j << ": EE, ix=" << EcalID.ix() <<  ", iy=" << EcalID.iy() << ", energy=" << theRecHit->energy() );
		   DEBUG("                                  x=" << p.x()       <<  ",  y=" << p.y() );
	      } // end if
	   } // end if ECAL
       } // loop over RecHits
     } // loop over towers
}

void ECALRecHitAnalyzer::DumpGeometry()
{
 
}

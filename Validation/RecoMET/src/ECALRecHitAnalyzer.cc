#include "Validation/RecoMET/interface/ECALRecHitAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
// author: Bobby Scurlock, University of Florida
// first version 11/20/2006

#define DEBUG(X) { if (debug_) { cout << X << endl; } }

ECALRecHitAnalyzer::ECALRecHitAnalyzer(const edm::ParameterSet& iConfig)
{
  
  // Retrieve Information from the Configuration File
  geometryFile_      = iConfig.getUntrackedParameter<std::string>("GeometryFile");
  outputFile_        = iConfig.getUntrackedParameter<std::string>("OutputFile");
  ECALRecHitsLabel_  = iConfig.getParameter<std::string>("ECALRecHitsLabel");
  EBRecHitsLabel_  = iConfig.getParameter<std::string>("EBRecHitsLabel");
  EERecHitsLabel_    = iConfig.getParameter<std::string>("EERecHitsLabel");

  debug_             = iConfig.getParameter<bool>("Debug");

  if (outputFile_.size() > 0)
    edm::LogInfo("OutputInfo") << " MET/HCALRecHit Task histograms will be saved to '" << outputFile_.c_str() << "'";
  else edm::LogInfo("OutputInfo") << " MET/HCALRecHit Task histograms will NOT be saved";
 
}

void ECALRecHitAnalyzer::endJob() {

  // Store the DAQ Histograms
  if (outputFile_.size() > 0 && dbe_)
    dbe_->save(outputFile_);
} 

void ECALRecHitAnalyzer::beginJob(const edm::EventSetup& iSetup){
  CurrentEvent = -1;
  // Book the Histograms
  BookHistos();
  // Fill the geometry histograms
  FillGeometry(iSetup);
}

void ECALRecHitAnalyzer::BookHistos()
{
  // get ahold of back-end interface
  dbe_ = edm::Service<DQMStore>().operator->();
  
  if (dbe_) {

  // Book Geometry Histograms
 dbe_->setCurrentFolder("RecoMETV/METTask/ECAL/geometry");
  // ECAL barrel
  me["hEB_ieta_iphi_etaMap"] = dbe_->book2D("hEB_ieta_iphi_etaMap","", 171, -85, 86, 360, 1, 361);
  me["hEB_ieta_iphi_phiMap"] = dbe_->book2D("hEB_ieta_iphi_phiMap","", 171, -85, 86, 360, 1, 361);
  me["hEB_ieta_detaMap"] = dbe_->book1D("hEB_ieta_detaMap","", 171, -85, 86);
  me["hEB_ieta_dphiMap"] = dbe_->book1D("hEB_ieta_dphiMap","", 171, -85, 86);
  // ECAL +endcap
  me["hEEpZ_ix_iy_xMap"] = dbe_->book2D("hEEpZ_ix_iy_xMap","", 100,1,101, 100,1,101);
  me["hEEpZ_ix_iy_yMap"] = dbe_->book2D("hEEpZ_ix_iy_yMap","", 100,1,101, 100,1,101);
  me["hEEpZ_ix_iy_dxMap"] = dbe_->book2D("hEEpZ_ix_iy_dxMap","", 100,1,101, 100,1,101);  
  me["hEEpZ_ix_iy_dyMap"] = dbe_->book2D("hEEpZ_ix_iy_dyMap","", 100,1,101, 100,1,101);
  // ECAL -endcap
  me["hEEmZ_ix_iy_xMap"] = dbe_->book2D("hEEmZ_ix_iy_xMap","", 100,1,101, 100,1,101);
  me["hEEmZ_ix_iy_yMap"] = dbe_->book2D("hEEmZ_ix_iy_yMap","", 100,1,101, 100,1,101);
  me["hEEmZ_ix_iy_dxMap"] = dbe_->book2D("hEEmZ_ix_iy_dxMap","", 100,1,101, 100,1,101);  
  me["hEEmZ_ix_iy_dyMap"] = dbe_->book2D("hEEmZ_ix_iy_dyMap","", 100,1,101, 100,1,101);

  // Initialize bins for geometry to -999 because z = 0 is a valid entry 
  for (int i=1; i<=100; i++)
    for (int j=1; j<=100; j++)
      {
	me["hEEpZ_ix_iy_xMap"]->setBinContent(i,j,-999);
	me["hEEpZ_ix_iy_yMap"]->setBinContent(i,j,-999);
	me["hEEpZ_ix_iy_dxMap"]->setBinContent(i,j,-999);
	me["hEEpZ_ix_iy_dyMap"]->setBinContent(i,j,-999);

	me["hEEmZ_ix_iy_xMap"]->setBinContent(i,j,-999);
	me["hEEmZ_ix_iy_yMap"]->setBinContent(i,j,-999);
	me["hEEmZ_ix_iy_dxMap"]->setBinContent(i,j,-999);
	me["hEEmZ_ix_iy_dyMap"]->setBinContent(i,j,-999);
      }

  for (int i=1; i<=171; i++)
    {
      me["hEB_ieta_detaMap"]->setBinContent(i,-999);
      me["hEB_ieta_dphiMap"]->setBinContent(i,-999);
      for (int j=1; j<=360; j++)
	{
	  me["hEB_ieta_iphi_etaMap"]->setBinContent(i,j,-999);
	  me["hEB_ieta_iphi_phiMap"]->setBinContent(i,j,-999);
	}
    }

  // Book Data Histograms
  dbe_->setCurrentFolder("RecoMETV/METTask/ECAL/data");
  // Energy Histograms by logical index
  me["hEEpZ_energy_ix_iy"] = dbe_->book2D("hEEpZ_energy_ix_iy","", 100,1,101, 100,1,101);
  me["hEEmZ_energy_ix_iy"] = dbe_->book2D("hEEmZ_energy_ix_iy","", 100,1,101, 100,1,101);
  me["hEB_energy_ieta_iphi"] = dbe_->book2D("hEB_energy_ieta_iphi","", 171, -85, 86, 360, 1, 361);   
  // Occupancy Histograms by logical index
  me["hEEpZ_Occ_ix_iy"] = dbe_->book2D("hEEpZ_Occ_ix_iy","", 100,1,101, 100,1,101);  
  me["hEEmZ_Occ_ix_iy"] = dbe_->book2D("hEEmZ_Occ_ix_iy","", 100,1,101, 100,1,101);  
  me["hEB_Occ_ieta_iphi"] = dbe_->book2D("hEB_Occ_ieta_iphi","",171, -85, 86, 360, 1, 361);   
  }
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
    me["hEB_ieta_iphi_etaMap"]->setBinContent(Crystal_ieta+86, Crystal_iphi, Crystal_eta);
    me["hEB_ieta_iphi_phiMap"]->setBinContent(Crystal_ieta+86, Crystal_iphi, (Crystal_phi*180/M_PI) );
    
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
	me["hEEmZ_ix_iy_xMap"]->setBinContent(Crystal_ix, Crystal_iy, Crystal_x);
	me["hEEmZ_ix_iy_yMap"]->setBinContent(Crystal_ix, Crystal_iy, Crystal_y);
      }
    // ECAL +endcap
    if (Crystal_zside == 1)
      {
	me["hEEpZ_ix_iy_xMap"]->setBinContent(Crystal_ix, Crystal_iy, Crystal_x);
	me["hEEpZ_ix_iy_yMap"]->setBinContent(Crystal_ix, Crystal_iy, Crystal_y);
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
      
      double eta = me["hEB_ieta_iphi_etaMap"]->getBinContent(ieta_, 1);
      double etam1 = -999;
      
      if (ieta==1) 
	etam1 = me["hEB_ieta_iphi_etaMap"]->getBinContent(85, 1);
      else 
	etam1 = me["hEB_ieta_iphi_etaMap"]->getBinContent(ieta_ - 1, 1);

      //double phi = me["hEB_ieta_iphi_phiMap"]->getBinContent(ieta_, 1);
      double deta = fabs( eta - etam1 );
      double dphi = fabs( me["hEB_ieta_iphi_phiMap"]->getBinContent(ieta_, 1) - me["hEB_ieta_iphi_phiMap"]->getBinContent(ieta_, 2) );
          
      currentLowEdge_eta += deta;
      me["hEB_ieta_detaMap"]->setBinContent(ieta_, deta); // positive rings
      me["hEB_ieta_dphiMap"]->setBinContent(ieta_, dphi); // positive rings
      me["hEB_ieta_detaMap"]->setBinContent(86-ieta, deta); // negative rings
      me["hEB_ieta_dphiMap"]->setBinContent(86-ieta, dphi); // negative rings
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
  iEvent.getByLabel( ECALRecHitsLabel_, EBRecHitsLabel_, EBRecHits );
  iEvent.getByLabel( ECALRecHitsLabel_, EERecHitsLabel_, EERecHits );
  DEBUG( "Got ECALRecHits");

  edm::Handle<reco::CandidateCollection> to;
  iEvent.getByLabel( "caloTowers", to );
  const CandidateCollection *towers = (CandidateCollection *)to.product();
  reco::CandidateCollection::const_iterator tower = towers->begin();
  edm::Ref<CaloTowerCollection> towerRef = tower->get<CaloTowerRef>();
  const CaloTowerCollection *towerCollection = towerRef.product();
  CaloTowerCollection::const_iterator calotower = towerCollection->begin();
  DEBUG( "Got Towers");    

  edm::ESHandle<CaloGeometry> pG;
  iSetup.get<IdealGeometryRecord>().get(pG);
  const CaloGeometry cG = *pG;
  const CaloSubdetectorGeometry* EBgeom=cG.getSubdetectorGeometry(DetId::Ecal,1);
  const CaloSubdetectorGeometry* EEgeom=cG.getSubdetectorGeometry(DetId::Ecal,2);
  DEBUG( "Got Geometry");
  DEBUG( "tower size = " << towerCollection->size());
  

 EBRecHitCollection::const_iterator ebrechit;
  int nEBrechit = 0;
  for (ebrechit = EBRecHits->begin(); ebrechit != EBRecHits->end(); ebrechit++) {
    
    EBDetId det = ebrechit->id();
    double Energy = ebrechit->energy();
    Int_t ieta = det.ieta();
    Int_t iphi = det.iphi();
    
    if (Energy>0)
      {
	me["hEB_energy_ieta_iphi"]->Fill(ieta, iphi, Energy);
	me["hEB_Occ_ieta_iphi"]->Fill(ieta, iphi);
      }

  } // loop over EB

  EERecHitCollection::const_iterator eerechit;
  int nEErechit = 0;
  for (eerechit = EERecHits->begin(); eerechit != EERecHits->end(); eerechit++) {
    
    EEDetId det = eerechit->id();
    double Energy = eerechit->energy();
    Int_t ix = det.ix();
    Int_t iy = det.iy();
    int Crystal_zside = det.zside();
	
    if (Energy>0)
      {
	if (Crystal_zside == -1)
	  {
	    me["hEEmZ_energy_ix_iy"]->Fill(ix, iy, Energy);
	    me["hEEmZ_Occ_ix_iy"]->Fill(ix, iy);
	  }
	if (Crystal_zside == 1)
	  {
	    me["hEEpZ_energy_ix_iy"]->Fill(ix, iy, Energy);
	    me["hEEpZ_Occ_ix_iy"]->Fill(ix, iy);
	  }
      }
  } // loop over EE
} // loop over RecHits



void ECALRecHitAnalyzer::DumpGeometry()
{
 
}

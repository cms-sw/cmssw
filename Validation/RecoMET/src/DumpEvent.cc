#include "Validation/RecoMET/interface/DumpEvent.h"
// author: Bobby Scurlock, University of Florida
// first version 11/20/2006

#define DEBUG(X) { if (debug_) { cout << X << endl; } }

DumpEvent::DumpEvent(const edm::ParameterSet& iConfig)
{
  EnergyThreshold = iConfig.getParameter<double>("EnergyThreshold");
  theEvent = iConfig.getParameter<int>("theEvent");
  FirstEvent = iConfig.getParameter<int>("FirstEvent");
  LastEvent = iConfig.getParameter<int>("LastEvent");
  debug_ = iConfig.getParameter<bool>("Debug");
}

void DumpEvent::endJob() 
{
  //Write out the histogram files.
  m_DataFile->Write();
} 

void DumpEvent::beginJob(const edm::EventSetup& iSetup){
  CurrentEvent = -1;
  // Make the output files
  m_DataFile=new TFile("DumpEvent_data.root" ,"RECREATE");  
  // Book the Histograms
  BookHistos();
 }

void DumpEvent::BookHistos()
{
  // Book Data Histograms
  m_DataFile->cd();
  hElectron_eta = new TH1F("hElectron_eta", "", 100,0,100);
  hElectron_phi = new TH1F("hElectron_phi", "", 100,0,100);
  hElectron_energy = new TH1F("hElectron_energy", "", 100,0,100);
  
  hPhoton_eta = new TH1F("hPhoton_eta", "", 100,0,100);
  hPhoton_phi = new TH1F("hPhoton_phi", "", 100,0,100);
  hPhoton_energy = new TH1F("hPhoton_energy", "", 100,0,100);
  
  hJet_eta = new TH1F("hJet_eta", "", 100,0,100);
  hJet_phi = new TH1F("hJet_phi", "", 100,0,100);
  hJet_energy = new TH1F("hJet_energy", "", 100,0,100);
  hCaloTowerToJetMap_ieta_iphi = new TH2F("hCaloTowerToJetMap_ieta_iphi","",83,-41,42, 73,0,73);
 
  hMuon_eta = new TH1F("hMuon_eta", "", 100,0,100);
  hMuon_phi = new TH1F("hMuon_phi", "", 100,0,100);
  hMuon_pt = new TH1F("hMuon_pt", "", 100,0,100);
  
  hRecoMET_phi = new TH1F("hRecoMET_phi", "", 1,0,1);
  hRecoMET_MET = new TH1F("hRecoMET_MET", "", 1,0,1);
  hGenMET_phi = new TH1F("hGenMET_phi", "", 1,0,1); 

  // SuperCluster Histos
  hEEpZ_SC_ix_iy = new TH2F("hEEpZ_SC_ix_iy","", 100,1,101, 100,1,101);  
  hEEmZ_SC_ix_iy = new TH2F("hEEmZ_SC_ix_iy","", 100,1,101, 100,1,101);  
  hEB_SC_ieta_iphi = new TH2F("hEB_SC_ieta_iphi","",171, -85, 86, 360, 1, 361);   
}

void DumpEvent::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  CurrentEvent++;
  DEBUG( "Event: " << CurrentEvent);
  if (CurrentEvent == theEvent) 
    {
      WritePhotons(iEvent, iSetup);
      WriteElectrons(iEvent, iSetup);
      WriteJets(iEvent, iSetup);
      WriteMuons(iEvent, iSetup);
      WriteMET(iEvent, iSetup);
      WriteSCs(iEvent, iSetup);
    }
}

void DumpEvent::WriteElectrons(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  // Get Reco Electrons
  edm::Handle<ElectronCollection> ElectronHandle;
  iEvent.getByLabel("pixelMatchElectrons",ElectronHandle);
  const ElectronCollection *Electrons = ElectronHandle.product();
  ElectronCollection::const_iterator elec;
  DEBUG("***ELECTRONS***" );
  DEBUG("Event has " << Electrons->size() << " reconstructed electrons");
  int ele=0;
  for (elec = Electrons->begin(); elec != Electrons->end(); elec++) 
    {
      Float_t eta = elec->eta();
      Float_t phi = elec->phi();
      Float_t energy = elec->energy();
      hElectron_eta->SetBinContent(ele+1, eta);
      hElectron_phi->SetBinContent(ele+1, phi);
      hElectron_energy->SetBinContent(ele+1, energy);
      ele++;
      DEBUG( "Electron # " << ele << " : eta =" << eta << ", phi = " << phi << ", energy = " << energy );
    }
}

void DumpEvent::WritePhotons(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  // Get Reco Photons
  edm::Handle<PhotonCollection> PhotonHandle; 
  iEvent.getByLabel("photons",PhotonHandle);
  const PhotonCollection *Photons = PhotonHandle.product();
  PhotonCollection::const_iterator gamma; 
  DEBUG("***PHOTONS***");
  DEBUG("Event has " << Photons->size() << " reconstructed photons");
  int photon=0;
  for (gamma = Photons->begin(); gamma != Photons->end(); gamma++) 
    {
      Float_t eta = gamma->eta();
      Float_t phi = gamma->phi();
      Float_t energy = gamma->energy();
      hPhoton_eta->SetBinContent(photon+1, eta);
      hPhoton_phi->SetBinContent(photon+1, phi);
      hPhoton_energy->SetBinContent(photon+1, energy);
      photon++;
      DEBUG( "Photon # " << photon << " : eta =" << eta << ", phi = " << phi << ", energy = " << energy );
    }
}

void DumpEvent::WriteJets(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  // Jets
  edm::Handle<CaloJetCollection> JetHandle;
  iEvent.getByLabel( "iterativeCone5CaloJets", JetHandle );
  const CaloJetCollection *Jets = JetHandle.product();
  // Towers
  edm::Handle<CaloTowerCollection> caloTowers;
  iEvent.getByLabel( "towerMaker", caloTowers );
  
  CaloJetCollection::const_iterator jet; 
  DEBUG("***JETS***");
  DEBUG("Event has " << Jets->size() << " reconstructed jets");
  int njet = 0;
  for (jet = Jets->begin(); jet != Jets->end(); jet++)
    {
      Float_t eta = jet->eta();
      Float_t phi = jet->phi();
      Float_t energy = jet->energy();
      hJet_eta->SetBinContent(njet+1, eta);
      hJet_phi->SetBinContent(njet+1, phi);
      hJet_energy->SetBinContent(njet+1, energy);
      njet++;
      DEBUG( "Jet # " << njet << " : eta =" << eta << ", phi = " << phi << ", energy = " << energy );
      
     int nConstituents= jet->nConstituents();
     for (int i = 0; i <nConstituents ; i++)
       {
         const CaloTower& tower = *(jet->getConstituent (i));
         int ietaTower = tower.id().ieta();
         int iphiTower = tower.id().iphi();
         hCaloTowerToJetMap_ieta_iphi->Fill(ietaTower, iphiTower, njet);
       } 

    }
}


void DumpEvent::WriteMuons(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  // Get Reco Muons
  edm::Handle<MuonCollection> MuonHandle; 
  iEvent.getByLabel("globalMuons",MuonHandle);
  const MuonCollection *Muons = MuonHandle.product();
  MuonCollection::const_iterator muon; 
  DEBUG("***MUONS***");
  DEBUG("Event has " << Muons->size() << " reconstructed muons");
  int mu=0;
  for (muon = Muons->begin(); muon != Muons->end(); muon++) 
    {
      Float_t eta = muon->eta();
      Float_t phi = muon->phi();
      Float_t pt = muon->pt();
      hMuon_eta->SetBinContent(mu+1, eta);
      hMuon_phi->SetBinContent(mu+1, phi);
      hMuon_pt->SetBinContent(mu+1, pt);
      mu++;
      DEBUG( "Muon # " << mu << " : eta =" << eta << ", phi = " << phi << ", Pt = " << pt );
    }
}


void DumpEvent::WriteMET(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{ 
  {
    // Reco MET
    edm::Handle<CaloMETCollection> metHandle;
    iEvent.getByLabel("met", metHandle);
    const CaloMETCollection *MET = metHandle.product();
    const CaloMET caloMET = MET->front();
    Float_t phi = caloMET.phi();
    Float_t ET = caloMET.pt();
    hRecoMET_phi->SetBinContent(1, phi);
    hRecoMET_MET->SetBinContent(1, ET);
    DEBUG("***RecoMET***");
    DEBUG("RecoMET: ET =" << ET << ", phi = " << phi);
  }
  
  {
    // Get GenMET objects from event
    edm::Handle<GenMETCollection> genmetHandle;
    iEvent.getByLabel("genMet", genmetHandle);
    const GenMETCollection *gmet = genmetHandle.product();
    const GenMET genMET = gmet->front();
    Float_t phi = genMET.phi();
    Float_t ET = genMET.pt();
    hGenMET_phi->SetBinContent(1, phi);
    DEBUG("***GenMET***");
    DEBUG("GenMET: MET = " << ET << ", phi = " << phi);
  }
}
  

void DumpEvent::WriteSCs(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  // Get ECAL Barrel identified SuperClusters
  edm::Handle<SuperClusterCollection> BSCHandle;
  iEvent.getByLabel("correctedHybridSuperClusters",BSCHandle);
  const SuperClusterCollection *BSuperClusters = BSCHandle.product();
  DEBUG("***Barrel Super Clusters***");
  Int_t whichSC = 0;
  SuperClusterCollection::const_iterator BSCC;// int num = 0;
  DEBUG("Event has " << BSuperClusters->size() << " Barrel SuperClusters" );
  int Bscs=0;
  for (BSCC = BSuperClusters->begin(); BSCC != BSuperClusters->end(); BSCC++) {
    whichSC++;
    Bscs++;
    Float_t energy = BSCC->energy();
    DEBUG("SC# " << Bscs << " : eta = " << BSCC->eta() << ", phi = " << BSCC->phi() << ", Energy = " << energy );
    
    vector<DetId> crystals = BSCC->getHitsByDetId();
    vector<DetId>::const_iterator crystal;
    //int cry = 0;
    for (crystal = crystals.begin(); crystal != crystals.end(); crystal++) 
      {      
	// get the subdetector
	int SubDet = (*crystal).subdetId();
	//DetId::Detector DetNum=(*crystal).det();
	if (SubDet == 1) 
	  {
	    EBDetId EcalID = *crystal;
	    float ieta = EcalID.ieta();
	    float iphi = EcalID.iphi();
	    hEB_SC_ieta_iphi->Fill(ieta, iphi, whichSC);
	  }
      }
  }
  
  // Get ECAL EndCap identified SuperClusters
  edm::Handle<SuperClusterCollection> ESCHandle;
  iEvent.getByLabel("correctedIslandEndcapSuperClusters",ESCHandle);
  const SuperClusterCollection *ESuperClusters = ESCHandle.product();
  DEBUG("***EndCap Super Clusters***");
  SuperClusterCollection::const_iterator ESCC;// int num = 0;
  DEBUG("Event has " << ESuperClusters->size() << " EndCap SuperClusters" );
  int Escs=0;
  for (ESCC = ESuperClusters->begin(); ESCC != ESuperClusters->end(); ESCC++) {
    whichSC++;
    Escs++;
    Float_t energy = ESCC->energy();
    DEBUG("SC# " << Bscs << " : eta = " << BSCC->eta() << ", phi = " << BSCC->phi() << ", Energy = " << energy );
    vector<DetId> crystals = ESCC->getHitsByDetId();
    vector<DetId>::const_iterator crystal;
    //int cry = 0;
    for (crystal = crystals.begin(); crystal != crystals.end(); crystal++) 
      {      
	// get the subdetector
	int SubDet = (*crystal).subdetId();
	//DetId::Detector DetNum=(*crystal).det();
	if (SubDet == 2) 
	  {
	    EEDetId EcalID = *crystal;
	    float ix = EcalID.ix();
	    float iy = EcalID.iy();
	    float Crystal_zside = EcalID.zside();
	    if (Crystal_zside == -1)
	      {
		hEEmZ_SC_ix_iy->Fill(ix, iy, whichSC);
	      }
	    if (Crystal_zside == 1)
	      {
		hEEpZ_SC_ix_iy->Fill(ix, iy, whichSC);
	      }
	  }     
    } // end loop over crystals
  }
}

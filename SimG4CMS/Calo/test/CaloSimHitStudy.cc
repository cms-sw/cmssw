#include "SimG4CMS/Calo/test/CaloSimHitStudy.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "SimG4CMS/Calo/interface/CaloHitID.h"

#include "FWCore/Utilities/interface/Exception.h"

CaloSimHitStudy::CaloSimHitStudy(const edm::ParameterSet& ps) {

  tok_evt_ = consumes<edm::HepMCProduct>(edm::InputTag(ps.getUntrackedParameter<std::string>("SourceLabel","VtxSmeared")));
  g4Label   = ps.getUntrackedParameter<std::string>("ModuleLabel","g4SimHits");
  hitLab[0] = ps.getUntrackedParameter<std::string>("EBCollection","EcalHitsEB");
  hitLab[1] = ps.getUntrackedParameter<std::string>("EECollection","EcalHitsEE");
  hitLab[2] = ps.getUntrackedParameter<std::string>("ESCollection","EcalHitsES");
  hitLab[3] = ps.getUntrackedParameter<std::string>("HCCollection","HcalHits");


  double maxEnergy_= ps.getUntrackedParameter<double>("MaxEnergy", 200.0);
  tmax_     = ps.getUntrackedParameter<double>("TimeCut", 100.0);
  eMIP_     = ps.getUntrackedParameter<double>("MIPCut",  0.70);

  muonLab[0]  = "MuonRPCHits";
  muonLab[1]  = "MuonCSCHits";
  muonLab[2]  = "MuonDTHits";
  tkHighLab[0]= "TrackerHitsPixelBarrelHighTof";
  tkHighLab[1]= "TrackerHitsPixelEndcapHighTof";
  tkHighLab[2]= "TrackerHitsTECHighTof";
  tkHighLab[3]= "TrackerHitsTIBHighTof";
  tkHighLab[4]= "TrackerHitsTIDHighTof";
  tkHighLab[5]= "TrackerHitsTOBHighTof";
  tkLowLab[0] = "TrackerHitsPixelBarrelLowTof";
  tkLowLab[1] = "TrackerHitsPixelEndcapLowTof";
  tkLowLab[2] = "TrackerHitsTECLowTof";
  tkLowLab[3] = "TrackerHitsTIBLowTof";
  tkLowLab[4] = "TrackerHitsTIDLowTof";
  tkLowLab[5] = "TrackerHitsTOBLowTof";

  // register for data access
  for ( unsigned i=0; i != 4; i++ )
    toks_calo_[i] = consumes<edm::PCaloHitContainer>(edm::InputTag(g4Label,hitLab[i]));

  for ( unsigned i=0; i != 3; i++ )
    toks_track_[i] = consumes<edm::PSimHitContainer>(edm::InputTag(g4Label,muonLab[i]));

  for ( unsigned i=0; i != 6; i++ ) {
    toks_tkHigh_[i] = consumes<edm::PSimHitContainer>(edm::InputTag(g4Label,tkHighLab[i]));
    toks_tkLow_[i] = consumes<edm::PSimHitContainer>(edm::InputTag(g4Label,tkLowLab[i]));
  }


  edm::LogInfo("HitStudy") << "Module Label: " << g4Label << "   Hits: "
			   << hitLab[0] << ", " << hitLab[1] << ", " 
			   << hitLab[2] << ", "<< hitLab[3] 
			   << "   MaxEnergy: " << maxEnergy_ << "  Tmax: "
			   << tmax_ << "  MIP Cut: " << eMIP_;

  edm::Service<TFileService> tfile;
 
  if ( !tfile.isAvailable() )
    throw cms::Exception("BadConfig") << "TFileService unavailable: "
                                      << "please add it to config file";
  char  name[20], title[120];
  sprintf (title, "Incident PT (GeV)");
  ptInc_ = tfile->make<TH1F>("PtInc", title, 1000, 0., maxEnergy_);
  ptInc_->GetXaxis()-> SetTitle(title);
  ptInc_->GetYaxis()->SetTitle("Events");
  sprintf (title, "Incident Energy (GeV)");
  eneInc_ = tfile->make<TH1F>("EneInc", title, 1000, 0., maxEnergy_);
  eneInc_->GetXaxis()-> SetTitle(title);
  eneInc_->GetYaxis()->SetTitle("Events");
  sprintf (title, "Incident #eta");
  etaInc_ = tfile->make<TH1F>("EtaInc", title, 200, -5., 5.);
  etaInc_->GetXaxis()->SetTitle(title);
  etaInc_->GetYaxis()->SetTitle("Events");
  sprintf (title, "Incident #phi");
  phiInc_ = tfile->make<TH1F>("PhiInc", title, 200, -3.1415926, 3.1415926);
  phiInc_->GetXaxis()->SetTitle(title);
  phiInc_->GetYaxis()->SetTitle("Events");
  std::string dets[9] = {"EB", "EB(APD)", "EB(ATJ)", "EE", "ES", "HB", "HE", "HO", "HF"};
  for (int i=0; i<9; i++) {
    sprintf (name, "Hit%d", i);
    sprintf (title, "Number of hits in %s", dets[i].c_str());
    hit_[i]  = tfile->make<TH1F>(name, title, 1000, 0., 20000.);
    hit_[i]->GetXaxis()->SetTitle(title); 
    hit_[i]->GetYaxis()->SetTitle("Events");
    sprintf (name, "Time%d", i);
    sprintf (title, "Time of the hit (ns) in %s", dets[i].c_str());
    time_[i]  = tfile->make<TH1F>(name, title, 1000, 0., 1000.);
    time_[i]->GetXaxis()->SetTitle(title); 
    time_[i]->GetYaxis()->SetTitle("Hits");
    sprintf (name, "TimeAll%d", i);
    sprintf (title, "Hit time (ns) in %s with no check on Track ID", dets[i].c_str());
    timeAll_[i]  = tfile->make<TH1F>(name, title, 1000, 0., 1000.);
    timeAll_[i]->GetXaxis()->SetTitle(title); 
    timeAll_[i]->GetYaxis()->SetTitle("Hits");
    double ymax = maxEnergy_;
    if (i==1 || i==2 || i==4) ymax = 1.0;
    else if (i>4 && i<8)      ymax = 10.0;
    sprintf (name, "Edep%d", i);
    sprintf (title, "Energy deposit (GeV) in %s", dets[i].c_str());
    edep_[i]  = tfile->make<TH1F>(name, title, 5000, 0., ymax);
    edep_[i]->GetXaxis()->SetTitle(title); 
    edep_[i]->GetYaxis()->SetTitle("Hits");
    sprintf (name, "EdepEM%d", i);
    sprintf (title, "Energy deposit (GeV) by EM particles in %s", dets[i].c_str());
    edepEM_[i]  = tfile->make<TH1F>(name, title, 5000, 0., ymax);
    edepEM_[i]->GetXaxis()->SetTitle(title); 
    edepEM_[i]->GetYaxis()->SetTitle("Hits");
    sprintf (name, "EdepHad%d", i);
    sprintf (title, "Energy deposit (GeV) by hadrons in %s", dets[i].c_str());
    edepHad_[i]  = tfile->make<TH1F>(name, title, 5000, 0., ymax);
    edepHad_[i]->GetXaxis()->SetTitle(title); 
    edepHad_[i]->GetYaxis()->SetTitle("Hits");
    sprintf (name, "Etot%d", i);
    sprintf (title, "Total energy deposit (GeV) in %s", dets[i].c_str());
    etot_[i]  = tfile->make<TH1F>(name, title, 5000, 0., ymax);
    etot_[i]->GetXaxis()->SetTitle(title); 
    etot_[i]->GetYaxis()->SetTitle("Events");
    sprintf (name, "EtotG%d", i);
    sprintf (title, "Total energy deposit (GeV) in %s (t < 100 ns)", dets[i].c_str());
    etotg_[i]  = tfile->make<TH1F>(name, title, 5000, 0., ymax);
    etotg_[i]->GetXaxis()->SetTitle(title); 
    etotg_[i]->GetYaxis()->SetTitle("Events");
  }
  std::string detx[9] = {"EB/EE (MIP)", "HB/HE (MIP)", "HB/HE/HO (MIP)", "EB/EE (no MIP)", "HB/HE (no MIP)", "HB/HE/HO (no MIP)", "EB/EE (All)", "HB/HE (All)", "HB/HE/HO (All)"};
  for (int i=0; i<9; i++) {
    double ymax = 1.0;
    if (i==0 || i==3 || i==6) ymax = maxEnergy_;
    sprintf (name, "EdepCal%d", i);
    sprintf (title, "Energy deposit in %s", detx[i].c_str());
    edepC_[i]  = tfile->make<TH1F>(name, title, 5000, 0., ymax);
    edepC_[i]->GetXaxis()->SetTitle(title); 
    edepC_[i]->GetYaxis()->SetTitle("Events");
    sprintf (name, "EdepCalT%d", i);
    sprintf (title, "Energy deposit (t < %f ns) in %s", tmax_,detx[i].c_str());
    edepT_[i]  = tfile->make<TH1F>(name, title, 5000, 0., ymax);
    edepT_[i]->GetXaxis()->SetTitle(title); 
    edepT_[i]->GetYaxis()->SetTitle("Events");
  }
  hitLow = tfile->make<TH1F>("HitLow","Number of hits in Track (Low)",1000,0,10000.);
  hitLow->GetXaxis()->SetTitle("Number of hits in Track (Low)");
  hitLow->GetYaxis()->SetTitle("Events");
  hitHigh= tfile->make<TH1F>("HitHigh","Number of hits in Track (High)",1000,0,10000.);
  hitHigh->GetXaxis()->SetTitle("Number of hits in Track (High)");
  hitHigh->GetYaxis()->SetTitle("Events");
  hitMu  = tfile->make<TH1F>("HitMu","Number of hits in Track (Muon)",1000,0,5000.);
  hitMu->GetXaxis()->SetTitle("Number of hits in Muon");
  hitMu->GetYaxis()->SetTitle("Events");
  std::string dett[15] = {"Pixel Barrel (High)", "Pixel Endcap (High)", 
			  "TEC (High)", "TIB (High)", "TID (High)", 
			  "TOB (High)", "Pixel Barrel (Low)", 
			  "Pixel Endcap (Low)","TEC (Low)", "TIB (Low)", 
			  "TID (Low)", "TOB (Low)", "RPC", "CSC", "DT"};
  for (int i=0; i<15; i++) {
    sprintf (name, "HitTk%d", i);
    sprintf (title, "Number of hits in %s", dett[i].c_str());
    hitTk_[i]  = tfile->make<TH1F>(name, title, 1000, 0., 1000.);
    hitTk_[i]->GetXaxis()->SetTitle(title); 
    hitTk_[i]->GetYaxis()->SetTitle("Events");
    sprintf (name, "TimeTk%d", i);
    sprintf (title, "Time of the hit (ns) in %s", dett[i].c_str());
    tofTk_[i]  = tfile->make<TH1F>(name, title, 1000, 0., 200.);
    tofTk_[i]->GetXaxis()->SetTitle(title); 
    tofTk_[i]->GetYaxis()->SetTitle("Hits");
    sprintf (name, "EdepTk%d", i);
    sprintf (title, "Energy deposit (GeV) in %s", dett[i].c_str());
    edepTk_[i]  = tfile->make<TH1F>(name, title, 5000, 0., 10.);
    edepTk_[i]->GetXaxis()->SetTitle(title); 
    edepTk_[i]->GetYaxis()->SetTitle("Hits");
  }
}

void CaloSimHitStudy::analyze(const edm::Event& e, const edm::EventSetup& ) {

  LogDebug("HitStudy") << "Run = " << e.id().run() << " Event = " 
		       << e.id().event();

  edm::Handle<edm::HepMCProduct > EvtHandle;
  e.getByToken(tok_evt_, EvtHandle);
  const  HepMC::GenEvent* myGenEvent = EvtHandle->GetEvent();

  double eInc=0, etaInc=0, phiInc=0;
  HepMC::GenEvent::particle_const_iterator p=myGenEvent->particles_begin();
  if (p != myGenEvent->particles_end()) {
    eInc   = (*p)->momentum().e();
    etaInc = (*p)->momentum().eta();
    phiInc = (*p)->momentum().phi();
  }
  double ptInc = eInc/std::cosh(etaInc);
  ptInc_->Fill(ptInc);
  eneInc_->Fill(eInc);
  etaInc_->Fill(etaInc);
  phiInc_->Fill(phiInc);

  std::vector<PCaloHit> ebHits, eeHits, hcHits;
  for (int i=0; i<4; i++) {
    bool getHits = false;
    edm::Handle<edm::PCaloHitContainer> hitsCalo;
    e.getByToken(toks_calo_[i],hitsCalo); 
    if (hitsCalo.isValid()) getHits = true;
    LogDebug("HitStudy") << "HcalValidation: Input flags Hits " << getHits;

    if (getHits) {
      std::vector<PCaloHit> caloHits;
      caloHits.insert(caloHits.end(),hitsCalo->begin(),hitsCalo->end());
      if (i == 0) ebHits.insert(ebHits.end(),hitsCalo->begin(),hitsCalo->end());
      else if (i == 1) eeHits.insert(eeHits.end(),hitsCalo->begin(),hitsCalo->end());
      else if (i == 3) hcHits.insert(hcHits.end(),hitsCalo->begin(),hitsCalo->end());
      LogDebug("HitStudy") << "HcalValidation: Hit buffer " 
			   << caloHits.size(); 
      analyzeHits (caloHits, i);
    }
  }
  analyzeHits (ebHits, eeHits, hcHits);

  std::vector<PSimHit>               muonHits;
  edm::Handle<edm::PSimHitContainer> hitsTrack;
  for (int i=0; i<3; i++) {
    e.getByToken(toks_track_[i],hitsTrack); 
    if (hitsTrack.isValid()) {
      muonHits.insert(muonHits.end(),hitsTrack->begin(),hitsTrack->end());
      analyzeHits (hitsTrack, i+12);
    }
  }
  unsigned int nhmu = muonHits.size();
  hitMu->Fill(double(nhmu));
  std::vector<PSimHit>               tkHighHits;
  for (int i=0; i<6; i++) {
    e.getByToken(toks_tkHigh_[i],hitsTrack); 
    if (hitsTrack.isValid()) {
      tkHighHits.insert(tkHighHits.end(),hitsTrack->begin(),hitsTrack->end());
      analyzeHits (hitsTrack, i);
    }
  }
  unsigned int nhtkh = tkHighHits.size();
  hitHigh->Fill(double(nhtkh));
  std::vector<PSimHit>               tkLowHits;
  for (int i=0; i<6; i++) {
    e.getByToken(toks_tkLow_[i],hitsTrack); 
    if (hitsTrack.isValid()) {
      tkLowHits.insert(tkLowHits.end(),hitsTrack->begin(),hitsTrack->end());
      analyzeHits (hitsTrack, i+6);
    }
  }
  unsigned int nhtkl = tkLowHits.size();
  hitLow->Fill(double(nhtkl));
}

void CaloSimHitStudy::analyzeHits (std::vector<PCaloHit>& hits, int indx) {

  int nHit = hits.size();
  int nHB=0, nHE=0, nHO=0, nHF=0, nEB=0, nEBAPD=0, nEBATJ=0, nEE=0, nES=0, nBad=0;
  std::map<unsigned int,double> hitMap;
  std::vector<double> etot(9,0), etotG(9,0);
  for (int i=0; i<nHit; i++) {
    double edep      = hits[i].energy();
    double time      = hits[i].time();
    unsigned int id_ = hits[i].id();
    double edepEM    = hits[i].energyEM();
    double edepHad   = hits[i].energyHad();
    if (indx == 0) {
      if      (hits[i].depth()==1) id_ |= 0x20000;
      else if (hits[i].depth()==2) id_ |= 0x40000;
    }
    std::map<unsigned int,double>::const_iterator it = hitMap.find(id_);
    if (it == hitMap.end()) {
      hitMap.insert(std::pair<unsigned int,double>(id_,time));
    }
    int idx = -1;
    if (indx != 3) {
      if (indx == 0)  idx = hits[i].depth();
      else            idx = indx+2;
      time_[idx]->Fill(time);
      edep_[idx]->Fill(edep);
      edepEM_[idx]->Fill(edepEM);
      edepHad_[idx]->Fill(edepHad);
      if      (idx == 0) nEB++;
      else if (idx == 1) nEBAPD++;
      else if (idx == 2) nEBATJ++;
      else if (idx == 3) nEE++;
      else if (idx == 4) nES++;
      else               nBad++;
      if (indx >=0 && indx < 3) {
	etot[idx] += edep;
	if (time < 100) etotG[idx] += edep;
      }
    } else {
      HcalDetId id     = HcalDetId(id_);
      int subdet       = id.subdet();
      if      (subdet == static_cast<int>(HcalBarrel))  {idx = indx+2; nHB++;}
      else if (subdet == static_cast<int>(HcalEndcap))  {idx = indx+3; nHE++;}
      else if (subdet == static_cast<int>(HcalOuter))   {idx = indx+4; nHO++;}
      else if (subdet == static_cast<int>(HcalForward)) {idx = indx+5; nHF++;}
      if (idx > 0) {
	time_[idx]->Fill(time);
	edep_[idx]->Fill(edep);
	edepEM_[idx]->Fill(edepEM);
	edepHad_[idx]->Fill(edepHad);
	etot[idx] += edep;
	if (time < 100) etotG[idx] += edep;
      } else {
	nBad++;
      }
    }
  }
  if (indx < 3) {
    etot_[indx+2]->Fill(etot[indx+2]);
    etotg_[indx+2]->Fill(etotG[indx+2]);
    if (indx == 0) {
      etot_[indx]->Fill(etot[indx]);
      etotg_[indx]->Fill(etotG[indx]);
      etot_[indx+1]->Fill(etot[indx+1]);
      etotg_[indx+1]->Fill(etotG[indx+1]);
      hit_[indx]->Fill(double(nEB));
      hit_[indx+1]->Fill(double(nEBAPD));
      hit_[indx+2]->Fill(double(nEBATJ));
    } else {
      hit_[indx+2]->Fill(double(nHit));
    }
  } else if (indx == 3) {
    hit_[5]->Fill(double(nHB));
    hit_[6]->Fill(double(nHE));
    hit_[7]->Fill(double(nHO));
    hit_[8]->Fill(double(nHF));
    for (int idx=5; idx<9; idx++) {
      etot_[idx]->Fill(etot[idx]);
      etotg_[idx]->Fill(etotG[idx]);
    }
  }

  LogDebug("HitStudy") << "CaloSimHitStudy::analyzeHits: EB " << nEB << ", "
		       << nEBAPD << ", " << nEBATJ << " EE " << nEE << " ES " 
		       << nES << " HB " << nHB << " HE " << nHE << " HO " 
		       << nHO << " HF " << nHF << " Bad " << nBad << " All " 
		       << nHit << " Reduced " << hitMap.size();
  std::map<unsigned int,double>::const_iterator it = hitMap.begin();
  for (; it !=hitMap.end(); it++) {
    double time      = it->second;
    unsigned int id_ = (it->first);
    int idx          = -1;
    if (indx < 3) {
      if (indx == 0) {
	if      ((id_&0x20000) != 0) idx = indx+1;
	else if ((id_&0x40000) != 0) idx = indx+1;
	else                         idx = indx;
      } else {
	idx  = indx + 2;
      }
      if (idx >= 0 && idx < 5) timeAll_[idx]->Fill(time);
    } else if (indx == 3) {
      HcalDetId id     = HcalDetId(id_);
      int idx          = -1;
      int subdet       = id.subdet();
      if      (subdet == static_cast<int>(HcalBarrel))  {idx = indx+2;}
      else if (subdet == static_cast<int>(HcalEndcap))  {idx = indx+3;}
      else if (subdet == static_cast<int>(HcalOuter))   {idx = indx+4;}
      else if (subdet == static_cast<int>(HcalForward)) {idx = indx+5;}
      if (idx > 0) {
	timeAll_[idx]->Fill(time);
      }
    }
  }

}

void CaloSimHitStudy::analyzeHits (edm::Handle<edm::PSimHitContainer>& hits, 
				   int indx) {

  int nHit = 0;
  edm::PSimHitContainer::const_iterator ihit;
  std::string label(" ");
  if      (indx >= 0 && indx < 6) label = tkHighLab[indx];
  else if (indx >= 6 && indx <12) label = tkLowLab[indx-6];
  else if (indx >=12 && indx <15) label = muonLab[indx-12];
  for (ihit=hits->begin(); ihit!=hits->end(); ihit++) {
    edepTk_[indx]->Fill(ihit->energyLoss());
    tofTk_[indx]->Fill(ihit->timeOfFlight());
    nHit++;
  }
  hitTk_[indx]->Fill(float(nHit));
  LogDebug("HitStudy") << "CaloSimHitStudy::analyzeHits: for " << label 
		       << " Index "  << indx << " # of Hits " << nHit;
}

void CaloSimHitStudy::analyzeHits (std::vector<PCaloHit>& ebHits, 
				   std::vector<PCaloHit>& eeHits, 
				   std::vector<PCaloHit>& hcHits) {

  double edepEB=0, edepEBT = 0;
  for (unsigned int i=0; i<ebHits.size(); i++) {
    double edep      = ebHits[i].energy();
    double time      = ebHits[i].time();
    if  (ebHits[i].depth()==0) {
      edepEB += edep;
      if (time < tmax_) edepEBT += edep;
    }
  }
  double edepEE=0, edepEET = 0;
  for (unsigned int i=0; i<eeHits.size(); i++) {
    double edep      = eeHits[i].energy();
    double time      = eeHits[i].time();
    edepEE += edep;
    if (time < tmax_) edepEET += edep;
  }
  double edepH=0, edepHT = 0, edepHO=0, edepHOT=0;
  for (unsigned int i=0; i<hcHits.size(); i++) {
    double edep      = hcHits[i].energy();
    double time      = hcHits[i].time();
    HcalDetId id     = HcalDetId(hcHits[i].id());
    int subdet       = id.subdet();
    if (subdet == static_cast<int>(HcalBarrel) || 
	subdet == static_cast<int>(HcalEndcap)) {
      edepH += edep;
      if (time < tmax_) edepHT += edep;
    } else if (subdet == static_cast<int>(HcalOuter)) {
      edepHO += edep;
      if (time < tmax_) edepHOT += edep;
    }
  }
  double edepE  = edepEB+edepEE;
  double edepET = edepEBT+edepEET;
  double edepHC = edepH+edepHO;
  double edepHCT= edepHT+edepHOT;
  LogDebug("HitStudy") << "CaloSimHitStudy::energy in EB " << edepEB << " ("
		       << edepEBT << ") from " << ebHits.size() << " hits; "
		       << " energy in EE " << edepEE << " (" << edepEET 
		       << ") from " << eeHits.size() << " hits; energy in HC "
		       << edepH << ", " << edepHO << " (" << edepHT << ", "
		       << edepHOT <<") from " << hcHits.size() << " hits";

  edepC_[6]->Fill(edepE);  edepT_[6]->Fill(edepET);
  edepC_[7]->Fill(edepH);  edepT_[7]->Fill(edepHT);
  edepC_[8]->Fill(edepHC); edepT_[8]->Fill(edepHCT);
  if (edepE < eMIP_) {
    edepC_[0]->Fill(edepE); edepC_[1]->Fill(edepH); edepC_[2]->Fill(edepHC);
  } else {
    edepC_[3]->Fill(edepE); edepC_[4]->Fill(edepH); edepC_[5]->Fill(edepHC);
  }
  if (edepET < eMIP_) {
    edepT_[0]->Fill(edepET); edepT_[1]->Fill(edepHT); edepT_[2]->Fill(edepHCT);
  } else {
    edepT_[3]->Fill(edepET); edepT_[4]->Fill(edepHT); edepT_[5]->Fill(edepHCT);
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(CaloSimHitStudy);

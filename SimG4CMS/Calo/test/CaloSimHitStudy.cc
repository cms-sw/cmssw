#include "SimG4CMS/Calo/test/CaloSimHitStudy.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "SimG4CMS/Calo/interface/CaloHitID.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "FWCore/Utilities/interface/Exception.h"

CaloSimHitStudy::CaloSimHitStudy(const edm::ParameterSet& ps) {

  sourceLabel = ps.getUntrackedParameter<std::string>("SourceLabel","generator");
  g4Label   = ps.getUntrackedParameter<std::string>("ModuleLabel","g4SimHits");
  hitLab[0] = ps.getUntrackedParameter<std::string>("EBCollection","EcalHitsEB");
  hitLab[1] = ps.getUntrackedParameter<std::string>("EECollection","EcalHitsEE");
  hitLab[2] = ps.getUntrackedParameter<std::string>("ESCollection","EcalHitsES");
  hitLab[3] = ps.getUntrackedParameter<std::string>("HCCollection","HcalHits");
  double maxEnergy_= ps.getUntrackedParameter<double>("MaxEnergy", 200.0);
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
  tkLowLab[0] = "TrackerHitsPixelEndcapLowTof";
  tkLowLab[0] = "TrackerHitsTECLowTof";
  tkLowLab[0] = "TrackerHitsTIBLowTof";
  tkLowLab[0] = "TrackerHitsTIDLowTof";
  tkLowLab[0] = "TrackerHitsTOBLowTof";
  edm::LogInfo("HitStudy") << "Module Label: " << g4Label << "   Hits: "
			   << hitLab[0] << ", " << hitLab[1] << ", " 
			   << hitLab[2] << ", "<< hitLab[3] 
			   << "   MaxEnergy: " << maxEnergy_;

  edm::Service<TFileService> tfile;
 
  if ( !tfile.isAvailable() )
    throw cms::Exception("BadConfig") << "TFileService unavailable: "
                                      << "please add it to config file";
  std::string dets[7] = {"EB", "EE", "ES", "HB", "HE", "HO", "HF"};
  char  name[20], title[100];
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
  for (int i=0; i<7; i++) {
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
    if (i > 1) ymax *= 0.05;
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
  hitLow = tfile->make<TH1F>("HitLow","Number of hits in Track (Low)",1000,0,10000.);
  hitLow->GetXaxis()->SetTitle("Number of hits in Track (Low)");
  hitLow->GetYaxis()->SetTitle("Events");
  hitHigh= tfile->make<TH1F>("HitHigh","Number of hits in Track (High)",1000,0,10000.);
  hitHigh->GetXaxis()->SetTitle("Number of hits in Track (High)");
  hitHigh->GetYaxis()->SetTitle("Events");
  hitMu  = tfile->make<TH1F>("HitMu","Number of hits in Track (Muon)",1000,0,5000.);
  hitMu->GetXaxis()->SetTitle("Number of hits in Muon");
  hitMu->GetYaxis()->SetTitle("Events");
}

void CaloSimHitStudy::analyze(const edm::Event& e, const edm::EventSetup& ) {

  LogDebug("HitStudy") << "Run = " << e.id().run() << " Event = " 
		       << e.id().event();

  edm::Handle<edm::HepMCProduct > EvtHandle;
  e.getByLabel(sourceLabel, EvtHandle);
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

  for (int i=0; i<4; i++) {
    bool getHits = false;
    edm::Handle<edm::PCaloHitContainer> hitsCalo;
    e.getByLabel(g4Label,hitLab[i],hitsCalo); 
    if (hitsCalo.isValid()) getHits = true;
    LogDebug("HitStudy") << "HcalValidation: Input flags Hits " << getHits;

    if (getHits) {
      std::vector<PCaloHit> caloHits;
      caloHits.insert(caloHits.end(),hitsCalo->begin(),hitsCalo->end());
      LogDebug("HitStudy") << "HcalValidation: Hit buffer " 
			   << caloHits.size(); 
      analyzeHits (caloHits, i);
    }
  }

  std::vector<PSimHit>               muonHits;
  edm::Handle<edm::PSimHitContainer> hitsTrack;
  for (int i=0; i<3; i++) {
    e.getByLabel(g4Label,muonLab[i],hitsTrack); 
    if (hitsTrack.isValid())
      muonHits.insert(muonHits.end(),hitsTrack->begin(),hitsTrack->end());
  }
  unsigned int nhmu = muonHits.size();
  hitMu->Fill(double(nhmu));
  std::vector<PSimHit>               tkHighHits;
  for (int i=0; i<6; i++) {
    e.getByLabel(g4Label,tkHighLab[i],hitsTrack); 
    if (hitsTrack.isValid())
      tkHighHits.insert(tkHighHits.end(),hitsTrack->begin(),hitsTrack->end());
  }
  unsigned int nhtkh = tkHighHits.size();
  hitHigh->Fill(double(nhtkh));
  std::vector<PSimHit>               tkLowHits;
  for (int i=0; i<6; i++) {
    e.getByLabel(g4Label,tkLowLab[i],hitsTrack); 
    if (hitsTrack.isValid())
      tkLowHits.insert(tkLowHits.end(),hitsTrack->begin(),hitsTrack->end());
  }
  unsigned int nhtkl = tkLowHits.size();
  hitLow->Fill(double(nhtkl));
}

void CaloSimHitStudy::analyzeHits (std::vector<PCaloHit>& hits, int indx) {

  int nHit = hits.size();
  int nHB=0, nHE=0, nHO=0, nHF=0, nEB=0, nEE=0, nES=0, nBad=0;
  std::map<unsigned int,double> hitMap;
  std::vector<double> etot(7,0), etotG(7,0);
  for (int i=0; i<nHit; i++) {
    double edep      = hits[i].energy();
    double time      = hits[i].time();
    unsigned int id_ = hits[i].id();
    double edepEM    = hits[i].energyEM();
    double edepHad   = hits[i].energyHad();
    std::map<unsigned int,double>::const_iterator it = hitMap.find(id_);
    if (it == hitMap.end()) {
      hitMap.insert(std::pair<unsigned int,double>(id_,time));
    }
    if (indx != 3) {
      time_[indx]->Fill(time);
      edep_[indx]->Fill(edep);
      edepEM_[indx]->Fill(edepEM);
      edepHad_[indx]->Fill(edepHad);
      if      (indx == 0) nEB++;
      else if (indx == 1) nEE++;
      else if (indx == 2) nES++;
      else                nBad++;
      if (indx < 3) {
	etot[indx] += edep;
	if (time < 100) etotG[indx] += edep;
      }
    } else {
      HcalDetId id     = HcalDetId(id_);
      int idx          = -1;
      int subdet       = id.subdet();
      if      (subdet == static_cast<int>(HcalBarrel))  {idx = indx; nHB++;}
      else if (subdet == static_cast<int>(HcalEndcap))  {idx = indx+1; nHE++;}
      else if (subdet == static_cast<int>(HcalOuter))   {idx = indx+2; nHO++;}
      else if (subdet == static_cast<int>(HcalForward)) {idx = indx+3; nHF++;}
      if (idx > 0) {
	time_[idx]->Fill(time);
	edep_[idx]->Fill(edep);
	edepEM_[indx]->Fill(edepEM);
	edepHad_[indx]->Fill(edepHad);
	etot[idx] += edep;
	if (time < 100) etotG[idx] += edep;
      }
    }
  }
  if (indx != 3) {
    hit_[indx]->Fill(double(nHit));
    etot_[indx]->Fill(etot[indx]);
    etotg_[indx]->Fill(etotG[indx]);
  } else {
    hit_[3]->Fill(double(nHB));
    hit_[4]->Fill(double(nHE));
    hit_[5]->Fill(double(nHO));
    hit_[6]->Fill(double(nHF));
    for (int idx=3; idx<7; idx++) {
      etot_[idx]->Fill(etot[idx]);
      etotg_[idx]->Fill(etotG[idx]);
    }
  }

  LogDebug("HitStudy") << "CaloSimHitStudy::analyzeHits: EB " << nEB << " EE "
		       << nEE << " ES " << nES << " HB " << nHB << " HE " 
		       << nHE << " HO " << nHO << " HF " << nHF << " Bad " 
		       << nBad << " All " << nHit << " Reduced " 
		       << hitMap.size();
  std::map<unsigned int,double>::const_iterator it = hitMap.begin();
  for (; it !=hitMap.end(); it++) {
    double time      = it->second;
    unsigned int id_ = (it->first);
    if (indx != 3) {
      timeAll_[indx]->Fill(time);
    } else {
      HcalDetId id     = HcalDetId(id_);
      int idx          = -1;
      int subdet       = id.subdet();
      if      (subdet == static_cast<int>(HcalBarrel))  {idx = indx;}
      else if (subdet == static_cast<int>(HcalEndcap))  {idx = indx+1;}
      else if (subdet == static_cast<int>(HcalOuter))   {idx = indx+2;}
      else if (subdet == static_cast<int>(HcalForward)) {idx = indx+3;}
      if (idx > 0) {
	timeAll_[idx]->Fill(time);
      }
    }
  }

}

//define this as a plug-in
DEFINE_FWK_MODULE(CaloSimHitStudy);

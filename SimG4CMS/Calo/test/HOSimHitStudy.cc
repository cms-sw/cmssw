#include "SimG4CMS/Calo/test/HOSimHitStudy.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "SimG4CMS/Calo/interface/CaloHitID.h"
#include "SimG4CMS/Calo/interface/HcalTestNumberingScheme.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include <iostream>
#include <iomanip>

HOSimHitStudy::HOSimHitStudy(const edm::ParameterSet& ps) {

  tok_evt_ = consumes<edm::HepMCProduct>(edm::InputTag(ps.getUntrackedParameter<std::string>("SourceLabel","VtxSmeared")));
  g4Label   = ps.getUntrackedParameter<std::string>("ModuleLabel","g4SimHits");
  hitLab[0] = ps.getUntrackedParameter<std::string>("EBCollection","EcalHitsEB");
  hitLab[1] = ps.getUntrackedParameter<std::string>("HCCollection","HcalHits");

  for ( unsigned i=0; i != 2; i++ )
    toks_calo_[i] = consumes<edm::PCaloHitContainer>(edm::InputTag(g4Label,hitLab[i]));

  maxEnergy = ps.getUntrackedParameter<double>("MaxEnergy", 200.0);
  scaleEB   = ps.getUntrackedParameter<double>("ScaleEB", 1.0);
  scaleHB   = ps.getUntrackedParameter<double>("ScaleHB", 100.0);
  scaleHO   = ps.getUntrackedParameter<double>("ScaleHO", 2.0);
  tcut_     = ps.getUntrackedParameter<double>("TimeCut", 100.0);
  scheme_   = ps.getUntrackedParameter<bool>("TestNumbering", false);
  print_    = ps.getUntrackedParameter<bool>("PrintExcessEnergy", true);
  edm::LogInfo("HitStudy") << "Module Label: " << g4Label << "   Hits: "
			   << hitLab[0] << ", " << hitLab[1] 
			   << "   MaxEnergy: " << maxEnergy
			   << "   Scale factor for EB " << scaleEB 
			   << ", for HB " << scaleHB << " and for HO " 
			   << scaleHO << " time Cut " << tcut_;

  edm::Service<TFileService> tfile;
 
  if ( !tfile.isAvailable() )
    throw cms::Exception("BadConfig") << "TFileService unavailable: "
                                      << "please add it to config file";
  std::string dets[3] = {"EB", "HB", "HO"};
  char  name[60], title[100];
  double ymax = maxEnergy;
  sprintf (title, "Incident Energy (GeV)");
  eneInc_ = tfile->make<TH1F>("EneInc", title, 1000, 0., ymax);
  eneInc_->GetXaxis()->SetTitle(title); 
  eneInc_->GetYaxis()->SetTitle("Events");
  sprintf (title, "Incident #eta");
  etaInc_ = tfile->make<TH1F>("EtaInc", title, 200, -5., 5.);
  etaInc_->GetXaxis()->SetTitle(title); 
  etaInc_->GetYaxis()->SetTitle("Events");
  sprintf (title, "Incident #phi");
  phiInc_ = tfile->make<TH1F>("PhiInc", title, 200, -3.1415926, 3.1415926);
  phiInc_->GetXaxis()->SetTitle(title); 
  phiInc_->GetYaxis()->SetTitle("Events");
  int itcut = (int)(tcut_);
  for (int i=0; i<3; i++) {
    sprintf (name, "Hit%d", i);
    sprintf (title, "Number of hits in %s", dets[i].c_str());
    hit_[i]  = tfile->make<TH1F>(name, title, 1000, 0., 20000.);
    hit_[i]->GetXaxis()->SetTitle(title); 
    hit_[i]->GetYaxis()->SetTitle("Events");
    sprintf (name, "Time%d", i);
    sprintf (title, "Time of the hit (ns) in %s", dets[i].c_str());
    time_[i]  = tfile->make<TH1F>(name, title, 1200, 0., 1200.);
    time_[i]->GetXaxis()->SetTitle(title); 
    time_[i]->GetYaxis()->SetTitle("Hits");
    if (i > 0) ymax = 1.0;
    else       ymax = 50.0;
    sprintf (name, "Edep%d", i);
    sprintf (title, "Energy deposit (GeV) in %s", dets[i].c_str());
    edep_[i]  = tfile->make<TH1F>(name, title, 5000, 0., ymax);
    edep_[i]->GetXaxis()->SetTitle(title); 
    edep_[i]->GetYaxis()->SetTitle("Hits");
    sprintf (name, "EdepT%d", i);
    sprintf (title, "Energy deposit (GeV) in %s for t < %d ns", dets[i].c_str(), itcut);
    edepT_[i]  = tfile->make<TH1F>(name, title, 5000, 0., ymax);
    edepT_[i]->GetXaxis()->SetTitle(title); 
    edepT_[i]->GetYaxis()->SetTitle("Hits");
    sprintf (name, "HitTow%d", i);
    sprintf (title, "Number of towers with hits in %s", dets[i].c_str());
    hitTow_[i]  = tfile->make<TH1F>(name, title, 1000, 0., 20000.);
    hitTow_[i]->GetXaxis()->SetTitle(title); 
    hitTow_[i]->GetYaxis()->SetTitle("Events");
    if (i > 0) ymax = 0.05*maxEnergy;
    else       ymax = maxEnergy;
    sprintf (name, "EdepTW%d", i);
    sprintf (title, "Energy deposit (GeV) in %s Tower", dets[i].c_str());
    edepTW_[i]  = tfile->make<TH1F>(name, title, 5000, 0., ymax);
    edepTW_[i]->GetXaxis()->SetTitle(title); 
    edepTW_[i]->GetYaxis()->SetTitle("Towers");
    sprintf (name, "EdepTWT%d", i);
    sprintf (title, "Energy deposit (GeV) in %s Tower for t < %d ns", dets[i].c_str(), itcut);
    edepTWT_[i]  = tfile->make<TH1F>(name, title, 5000, 0., ymax);
    edepTWT_[i]->GetXaxis()->SetTitle(title); 
    edepTWT_[i]->GetYaxis()->SetTitle("Towers");
    sprintf (name, "EdepZone%d", i);
    sprintf (title, "Energy deposit (GeV) in %s", dets[i].c_str());
    edepZon_[i]  = tfile->make<TH1F>(name, title, 5000, 0., ymax);
    edepZon_[i]->GetXaxis()->SetTitle(title); 
    edepZon_[i]->GetYaxis()->SetTitle("Events");
    sprintf (name, "EdepZoneT%d", i);
    sprintf (title, "Energy deposit (GeV) in %s for t < %d ns", dets[i].c_str(), itcut);
    edepZonT_[i]  = tfile->make<TH1F>(name, title, 5000, 0., ymax);
    edepZonT_[i]->GetXaxis()->SetTitle(title); 
    edepZonT_[i]->GetYaxis()->SetTitle("Events");
  }
  sprintf (title, "Energy Measured in EB (GeV)");
  eEB_  = tfile->make<TH1F>("EEB", title, 5000, 0., maxEnergy);
  eEB_->GetXaxis()->SetTitle(title); 
  eEB_->GetYaxis()->SetTitle("Events");
  sprintf (title, "Energy Measured in EB+HB (GeV)");
  eEBHB_  = tfile->make<TH1F>("EEBHB", title, 5000, 0., maxEnergy);
  eEBHB_->GetXaxis()->SetTitle(title); 
  eEBHB_->GetYaxis()->SetTitle("Events");
  sprintf (title, "Energy Measured in EB+HB+HO (GeV)");
  eEBHBHO_  = tfile->make<TH1F>("EEBHBHO", title, 5000, 0., maxEnergy);
  eEBHBHO_->GetXaxis()->SetTitle(title); 
  eEBHBHO_->GetYaxis()->SetTitle("Events");
  sprintf (title, "Energy Measured in EB (GeV) for t < %d ns", itcut);
  eEBT_  = tfile->make<TH1F>("EEBT", title, 5000, 0., maxEnergy);
  eEBT_->GetXaxis()->SetTitle(title); 
  eEBT_->GetYaxis()->SetTitle("Events");
  sprintf (title, "Energy Measured in EB+HB (GeV) for t < %d ns", itcut);
  eEBHBT_  = tfile->make<TH1F>("EEBHBT", title, 5000, 0., maxEnergy);
  eEBHBT_->GetXaxis()->SetTitle(title); 
  eEBHBT_->GetYaxis()->SetTitle("Events");
  sprintf (title, "Energy Measured in EB+HB+HO (GeV) for t < %d ns", itcut);
  eEBHBHOT_  = tfile->make<TH1F>("EEBHBHOT", title, 5000, 0., maxEnergy);
  eEBHBHOT_->GetXaxis()->SetTitle(title); 
  eEBHBHOT_->GetYaxis()->SetTitle("Events");
  sprintf (title, "SimHit energy in HO");
  eHO1_ = tfile->make<TProfile>("EHO1", title, 30, -1.305, 1.305);
  eHO1_->GetXaxis()->SetTitle(title); 
  eHO1_->GetYaxis()->SetTitle("Events");
  eHO2_ = tfile->make<TProfile2D>("EHO2", title, 30,-1.305,1.305,72,-3.1415926,3.1415926);
  eHO2_->GetXaxis()->SetTitle(title); 
  eHO2_->GetYaxis()->SetTitle("Events");
  sprintf (title, "SimHit energy in HO Layer 17");
  eHO17_ = tfile->make<TProfile>("EHO17", title, 30, -1.305, 1.305);
  eHO17_->GetXaxis()->SetTitle(title); 
  eHO17_->GetYaxis()->SetTitle("Events");
  sprintf (title, "SimHit energy in HO Layer 18");
  eHO18_ = tfile->make<TProfile>("EHO18", title, 30, -1.305, 1.305);
  eHO18_->GetXaxis()->SetTitle(title); 
  eHO18_->GetYaxis()->SetTitle("Events");
  sprintf (title, "SimHit energy in HO for t < %d ns", itcut);
  eHO1T_ = tfile->make<TProfile>("EHO1T", title, 30, -1.305, 1.305);
  eHO1T_->GetXaxis()->SetTitle(title); 
  eHO1T_->GetYaxis()->SetTitle("Events");
  eHO2T_ = tfile->make<TProfile2D>("EHO2T", title, 30,-1.305,1.305,72,-3.1415926,3.1415926);
  eHO2T_->GetXaxis()->SetTitle(title); 
  eHO2T_->GetYaxis()->SetTitle("Events");
  sprintf (title, "SimHit energy in HO Layer 17 for t < %d ns", itcut);
  eHO17T_ = tfile->make<TProfile>("EHO17T", title, 30, -1.305, 1.305);
  eHO17T_->GetXaxis()->SetTitle(title); 
  eHO17T_->GetYaxis()->SetTitle("Events");
  sprintf (title, "SimHit energy in HO Layer 18 for t < %d ns", itcut);
  eHO18T_ = tfile->make<TProfile>("EHO18T", title, 30, -1.305, 1.305);
  eHO18T_->GetXaxis()->SetTitle(title); 
  eHO18T_->GetYaxis()->SetTitle("Events");
  sprintf (title, "Number of layers hit in HO");
  nHO1_ = tfile->make<TProfile>("NHO1", title, 30, -1.305, 1.305);
  nHO1_->GetXaxis()->SetTitle(title); 
  nHO1_->GetYaxis()->SetTitle("Events");
  nHO2_ = tfile->make<TProfile2D>("NHO2", title, 30,-1.305,1.305,72,-3.1415926,3.1415926);
  nHO2_->GetXaxis()->SetTitle(title); 
  nHO2_->GetYaxis()->SetTitle("Events");
  sprintf (title, "Number of layers hit in HO for t < %d ns", itcut);
  nHO1T_ = tfile->make<TProfile>("NHO1T", title, 30, -1.305, 1.305);
  nHO1T_->GetXaxis()->SetTitle(title); 
  nHO1T_->GetYaxis()->SetTitle("Events");
  nHO2T_ = tfile->make<TProfile2D>("NHO2T", title, 30,-1.305,1.305,72,-3.1415926,3.1415926);
  nHO2T_->GetXaxis()->SetTitle(title); 
  nHO2T_->GetYaxis()->SetTitle("Events");
  for (int i=0; i<15; i++) {
    sprintf (name, "EHOE%d", i+1);
    sprintf (title, "SimHit energy in HO (Beam in #eta=%d bin)",i+1);
    eHOE_[i] = tfile->make<TH1F>(name, title, 1000, 0., 0.25);
    eHOE_[i]->GetXaxis()->SetTitle(title); 
    eHOE_[i]->GetYaxis()->SetTitle("Events");
    sprintf (name, "EHOE17%d", i+1);
    sprintf (title, "SimHit energy in Layer 17 (Beam in #eta=%d bin)",i+1);
    eHOE17_[i] = tfile->make<TH1F>(name, title, 1000, 0., 0.25);
    eHOE17_[i]->GetXaxis()->SetTitle(title); 
    eHOE17_[i]->GetYaxis()->SetTitle("Events");
    sprintf (name, "EHOE18%d", i+1);
    sprintf (title, "SimHit energy in Layer 18 (Beam in #eta=%d bin)",i+1);
    eHOE18_[i] = tfile->make<TH1F>(name, title, 1000, 0., 0.25);
    eHOE18_[i]->GetXaxis()->SetTitle(title); 
    eHOE18_[i]->GetYaxis()->SetTitle("Events");
    sprintf (name, "EHOE%dT", i+1);
    sprintf (title, "SimHit energy in HO (Beam in #eta=%d bin, t < %d ns)",i+1,itcut);
    eHOET_[i] = tfile->make<TH1F>(name, title, 1000, 0., 0.25);
    eHOET_[i]->GetXaxis()->SetTitle(title); 
    eHOET_[i]->GetYaxis()->SetTitle("Events");
    sprintf (name, "EHOE17%dT", i+1);
    sprintf (title, "SimHit energy in Layer 17 (Beam in #eta=%d bin, t < %d ns)",i+1,itcut);
    eHOE17T_[i] = tfile->make<TH1F>(name, title, 1000, 0., 0.25);
    eHOE17T_[i]->GetXaxis()->SetTitle(title); 
    eHOE17T_[i]->GetYaxis()->SetTitle("Events");
    sprintf (name, "EHOE18%dT", i+1);
    sprintf (title, "SimHit energy in Layer 18 (Beam in #eta=%d bin, t < %d ns)",i+1,itcut);
    eHOE18T_[i] = tfile->make<TH1F>(name, title, 1000, 0., 0.25);
    eHOE18T_[i]->GetXaxis()->SetTitle(title); 
    eHOE18T_[i]->GetYaxis()->SetTitle("Events");
    sprintf (name, "EHOEta%d", i+1);
    sprintf (title, "SimHit energy in HO #eta bin %d (Beam in #eta=%d bin)",i+1,i+1);
    eHOEta_[i] = tfile->make<TH1F>(name, title, 1000, 0., 0.25);
    eHOEta_[i]->GetXaxis()->SetTitle(title); 
    eHOEta_[i]->GetYaxis()->SetTitle("Events");
    sprintf (name, "EHOEta17%d", i+1);
    sprintf (title, "SimHit energy in Layer 17 #eta bin %d (Beam in #eta=%d bin)",i+1,i+1);
    eHOEta17_[i] = tfile->make<TH1F>(name, title, 1000, 0., 0.25);
    eHOEta17_[i]->GetXaxis()->SetTitle(title); 
    eHOEta17_[i]->GetYaxis()->SetTitle("Events");
    sprintf (name, "EHOEta18%d", i+1);
    sprintf (title, "SimHit energy in Layer 18 #eta bin %d (Beam in #eta=%d bin)",i+1,i+1);
    eHOEta18_[i] = tfile->make<TH1F>(name, title, 1000, 0., 0.25);
    eHOEta18_[i]->GetXaxis()->SetTitle(title); 
    eHOEta18_[i]->GetYaxis()->SetTitle("Events");
    sprintf (name, "EHOEta%dT", i+1);
    sprintf (title, "SimHit energy in HO #eta bin %d (Beam in #eta=%d bin, t < %d ns)",i+1,i+1,itcut);
    eHOEtaT_[i] = tfile->make<TH1F>(name, title, 1000, 0., 0.25);
    eHOEtaT_[i]->GetXaxis()->SetTitle(title); 
    eHOEtaT_[i]->GetYaxis()->SetTitle("Events");
    sprintf (name, "EHOEta17%dT", i+1);
    sprintf (title, "SimHit energy in Layer 17 #eta bin %d (Beam in #eta=%d bin, t < %d ns)",i+1,i+1,itcut);
    eHOEta17T_[i] = tfile->make<TH1F>(name, title, 1000, 0., 0.25);
    eHOEta17T_[i]->GetXaxis()->SetTitle(title); 
    eHOEta17T_[i]->GetYaxis()->SetTitle("Events");
    sprintf (name, "EHOEta18%dT", i+1);
    sprintf (title, "SimHit energy in Layer 18 #eta bin %d (Beam in #eta=%d bin, t < %d ns)",i+1,i+1,itcut);
    eHOEta18T_[i] = tfile->make<TH1F>(name, title, 1000, 0., 0.25);
    eHOEta18T_[i]->GetXaxis()->SetTitle(title); 
    eHOEta18T_[i]->GetYaxis()->SetTitle("Events");
    sprintf (name, "NHOE1%d", i+1);
    sprintf (title, "Number of layers hit in HO  (Beam in #eta=%d bin)",i+1);
    nHOE1_[i] = tfile->make<TH1F>(name, title, 20, 0, 20.);
    nHOE1_[i]->GetXaxis()->SetTitle(title); 
    nHOE1_[i]->GetYaxis()->SetTitle("Events");
    sprintf (name, "NHOE2%d", i+1);
    nHOE2_[i] = tfile->make<TProfile>(name, title, 72, -3.1415926, 3.1415926);
    nHOE2_[i]->GetXaxis()->SetTitle(title); 
    nHOE2_[i]->GetYaxis()->SetTitle("Events");
    sprintf (name, "NHOE1%dT", i+1);
    sprintf (title, "Number of layers hit in HO (Beam in #eta=%d bin, t < %d ns)", i+1,itcut);
    nHOE1T_[i] = tfile->make<TH1F>(name, title, 20, 0, 20.);
    nHOE1T_[i]->GetXaxis()->SetTitle(title); 
    nHOE1T_[i]->GetYaxis()->SetTitle("Events");
    sprintf (name, "NHOE2%dT", i+1);
    nHOE2T_[i] = tfile->make<TProfile>(name, title, 72, -3.1415926, 3.1415926);
    nHOE2T_[i]->GetXaxis()->SetTitle(title); 
    nHOE2T_[i]->GetYaxis()->SetTitle("Events");
    sprintf (name, "NHOEta1%d", i+1);
    sprintf (title, "Number of layers hit in HO #eta bin %d (Beam in #eta=%d bin)", i+1,i+1);
    nHOEta1_[i] = tfile->make<TH1F>(name, title, 20, 0, 20.);
    nHOEta1_[i]->GetXaxis()->SetTitle(title); 
    nHOEta1_[i]->GetYaxis()->SetTitle("Events");
    sprintf (name, "NHOEta2%d", i+1);
    nHOEta2_[i] = tfile->make<TProfile>(name, title, 72, -3.1415926, 3.1415926);
    nHOEta2_[i]->GetXaxis()->SetTitle(title); 
    nHOEta2_[i]->GetYaxis()->SetTitle("Events");
    sprintf (name, "NHOEta1%dT", i+1);
    sprintf (title, "Number of layers hit in HO #eta bin %d (Beam in #eta=%d bin, t < %d ns)", i+1,i+1,itcut);
    nHOEta1T_[i] = tfile->make<TH1F>(name, title, 20, 0, 20.);
    nHOEta1T_[i]->GetXaxis()->SetTitle(title); 
    nHOEta1T_[i]->GetYaxis()->SetTitle("Events");
    sprintf (name, "NHOEta2%dT", i+1);
    nHOEta2T_[i] = tfile->make<TProfile>(name, title, 72, -3.1415926, 3.1415926);
    nHOEta2T_[i]->GetXaxis()->SetTitle(title); 
    nHOEta2T_[i]->GetYaxis()->SetTitle("Events");
  }
}

void HOSimHitStudy::analyze(const edm::Event& e, const edm::EventSetup& ) {

  LogDebug("HitStudy") << "Run = " << e.id().run() << " Event = " 
		       << e.id().event();

  edm::Handle<edm::HepMCProduct > EvtHandle;
  e.getByToken(tok_evt_, EvtHandle);
  const  HepMC::GenEvent* myGenEvent = EvtHandle->GetEvent();

  eInc = etaInc = phiInc = 0;
  HepMC::GenEvent::particle_const_iterator p=myGenEvent->particles_begin();
  if (p != myGenEvent->particles_end()) {
    eInc   = (*p)->momentum().e();
    etaInc = (*p)->momentum().eta();
    phiInc = (*p)->momentum().phi();
  }

  LogDebug("HitStudy") << "Energy = " << eInc << " Eta = " << etaInc 
		       << " Phi = " << phiInc/CLHEP::deg;

  for (int i=0; i<2; i++) {
    bool getHits = false;
    if (i == 0) ecalHits.clear();
    else        hcalHits.clear();
    edm::Handle<edm::PCaloHitContainer> hitsCalo;
    e.getByToken(toks_calo_[i],hitsCalo); 
    if (hitsCalo.isValid()) getHits = true;
    LogDebug("HitStudy") << "HcalValidation: Input flag " << hitLab[i] 
			 << " getHits flag " << getHits;

    if (getHits) {
      unsigned int isiz;
      if (i == 0) {
	ecalHits.insert(ecalHits.end(),hitsCalo->begin(),hitsCalo->end());
	isiz = ecalHits.size();
      } else {
	hcalHits.insert(hcalHits.end(),hitsCalo->begin(),hitsCalo->end());
	isiz = hcalHits.size();
      }
      LogDebug("HitStudy") << "HcalValidation: Hit buffer for " << hitLab[i]
			   << " has " << isiz << " hits"; 
    }
  }
  analyzeHits ();
}

void HOSimHitStudy::analyzeHits () {

  //initialize
  int    nhit[3];
  double etot[3], etotT[3];
  std::vector<unsigned int> ebID, hbID, hoID;
  std::vector<double>       ebEtow,  hbEtow,  hoEtow;
  std::vector<double>       ebEtowT, hbEtowT, hoEtowT;
  for (int k=0; k<3; k++) {
    nhit[k] = 0;
    etot[k] = 0;
    etotT[k]= 0;
  }
  eneInc_->Fill(eInc);
  etaInc_->Fill(etaInc);
  phiInc_->Fill(phiInc);
  
  double eHO17=0, eHO18=0, eHO17T=0, eHO18T=0;
  double eHOE17[15], eHOE18[15], eHOE17T[15], eHOE18T[15];
  for (int k=0; k<15; k++) {
    eHOE17[k] = eHOE18[k] = eHOE17T[k] = eHOE18T[k] = 0;
  }
  // Loop over containers
  for (int k=0; k<2; k++) {
    int nHit;
    if (k == 0) {
      nHit = ecalHits.size();
    } else {
      nHit = hcalHits.size();
    }
    for (int i=0; i<nHit; i++) {
      double       edep, time;
      int          indx=-1;
      unsigned int id_;
      if (k == 0) {
	indx = 0;
	edep = ecalHits[i].energy();
	time = ecalHits[i].time();
	id_  = ecalHits[i].id();
      } else {
	edep = hcalHits[i].energy();
	time = hcalHits[i].time();
	id_  = hcalHits[i].id();
	int subdet, zside, depth, eta, phi, lay;
	if (scheme_) {
	  HcalTestNumberingScheme::unpackHcalIndex(id_, subdet, zside, depth, eta, phi, lay);
	} else {
	  HcalDetId id = HcalDetId(id_);
	  subdet       = id.subdet();
	  zside        = id.zside();
	  depth        = id.depth();
          eta          = id.ietaAbs();
          phi          = id.iphi();
          lay          = -1;
	}
	LogDebug("HitStudy") << "HcalValidation:: Hit " << k << " Subdet:"
			     << subdet << " zside:" << zside << " depth:" 
			     << depth << " layer:" << lay << " eta:" << eta
			     << " phi:" << phi;
	if      (subdet == static_cast<int>(HcalBarrel))  indx = 1;
	else if (subdet == static_cast<int>(HcalOuter)) {
	  indx = 2;
	  if (lay == 18)       {
	    eHO17 += edep;
	    if (time < tcut_) eHO17T += edep;
	    if (eta >=0 && eta < 15) {
	      eHOE17[eta] += edep;
	      if (time < tcut_) eHOE17T[eta] += edep;
	    }
	  } else {
	    eHO18 += edep;
	    if (time < tcut_) eHO18T += edep;
	    if (eta >=0 && eta < 15) {
	      eHOE18[eta] += edep;
	      if (time < tcut_) eHOE18T[eta] += edep;
	    }
	  }
	}
      }
      if (indx >= 0) {
	double edepT = edep;
	time_[indx]->Fill(time);
	edep_[indx]->Fill(edep);
	etot[indx] += edep;
	if (time < tcut_) {
	  etotT[indx] += edep;
	  edepT_[indx]->Fill(edep);
	  edepT = 0;
	}
	nhit[indx]++;
	if (indx == 0) {
	  bool ok = false;
	  for (unsigned int j=0; j<ebID.size(); j++) {
	    if (id_ == ebID[j]) {
	      ebEtow[j]  += edep;
	      ebEtowT[j] += edepT;
	      ok          = true;
	      break;
	    }
	  }
	  if (!ok) {
	    ebID.push_back(id_);
	    ebEtow.push_back(edep);
	    ebEtowT.push_back(edepT);
	  }
	} else if (indx == 1) {
	  bool ok = false;
	  for (unsigned int j=0; j<hbID.size(); j++) {
	    if (id_ == hbID[j]) {
	      hbEtow[j]  += edep;
	      hbEtowT[j] += edepT;
	      ok          = true;
	      break;
	    }
	  }
	  if (!ok) {
	    hbID.push_back(id_);
	    hbEtow.push_back(edep);
	    hbEtowT.push_back(edepT);
	  }
	} else {
	  bool ok = false;
	  for (unsigned int j=0; j<hoID.size(); j++) {
	    if (id_ == hoID[j]) {
	      hoEtow[j]  += edep;
	      hoEtowT[j] += edepT;
	      ok          = true;
	      break;
	    }
	  }
	  if (!ok) {
	    hoID.push_back(id_);
	    hoEtow.push_back(edep);
	    hoEtowT.push_back(edepT);
	  }
	}
      }
    }
  }

  // Now for towers and total energy deposits
  for (int k=0; k<3; k++) {
    hit_[k]->Fill(double(nhit[k]));
    edepZon_[k]->Fill(etot[k]);
    edepZonT_[k]->Fill(etotT[k]);
  }
  hitTow_[0]->Fill(double(ebEtow.size()));
  for (unsigned int i=0; i<ebEtow.size(); i++) {
    edepTW_[0]->Fill(ebEtow[i]);
    edepTWT_[0]->Fill(ebEtowT[i]);
  }
  hitTow_[1]->Fill(double(hbEtow.size()));
  for (unsigned int i=0; i<hbEtow.size(); i++) {
    edepTW_[1]->Fill(hbEtow[i]);
    edepTWT_[1]->Fill(hbEtowT[i]);
  }
  hitTow_[2]->Fill(double(hoEtow.size()));
  for (unsigned int i=0; i<hoEtow.size(); i++) {
    edepTW_[2]->Fill(hoEtow[i]);
    edepTWT_[2]->Fill(hoEtowT[i]);
  }
  double eEB     = scaleEB*etot[0];
  double eEBHB   = eEB + scaleHB*etot[1];
  double eEBHBHO = eEBHB + scaleHB*scaleHO*etot[2];
  eEB_->Fill(eEB);
  eEBHB_->Fill(eEBHB);
  eEBHBHO_->Fill(eEBHBHO);
  double eEBT     = scaleEB*etotT[0];
  double eEBHBT   = eEBT + scaleHB*etotT[1];
  double eEBHBHOT = eEBHBT + scaleHB*scaleHO*etotT[2];
  eEBT_->Fill(eEBT);
  eEBHBT_->Fill(eEBHBT);
  eEBHBHOT_->Fill(eEBHBHOT);
  eHO1_->Fill(etaInc,eHO17+eHO18);
  eHO2_->Fill(etaInc,phiInc,eHO17+eHO18);
  eHO17_->Fill(etaInc,eHO17);
  eHO18_->Fill(etaInc,eHO18);
  eHO1T_->Fill(etaInc,eHO17T+eHO18T);
  eHO2T_->Fill(etaInc,phiInc,eHO17T+eHO18T);
  eHO17T_->Fill(etaInc,eHO17T);
  eHO18T_->Fill(etaInc,eHO18T);
  int nHO=0, nHOT=0;
  if (eHO17 > 0) nHO++; if (eHO17T > 0) nHOT++;
  if (eHO18 > 0) nHO++; if (eHO18T > 0) nHOT++;
  nHO1_->Fill(etaInc,(double)(nHO));
  nHO2_->Fill(etaInc,phiInc,(double)(nHO));
  nHO1T_->Fill(etaInc,(double)(nHOT));
  nHO2T_->Fill(etaInc,phiInc,(double)(nHOT));
  int ieta=15;
  for (int k=0; k<15; ++k) {
    if (std::abs(etaInc) < 0.087*(k+1)) {
      ieta = k; break;
    }
  }
  if (ieta>=0 && ieta<15) {
    eHOE_[ieta]->Fill(eHO17+eHO18);
    eHOE17_[ieta]->Fill(eHO17);
    eHOE18_[ieta]->Fill(eHO18);
    eHOET_[ieta]->Fill(eHO17T+eHO18T);
    eHOE17T_[ieta]->Fill(eHO17T);
    eHOE18T_[ieta]->Fill(eHO18T);
    eHOEta_[ieta]->Fill(eHOE17[ieta]+eHOE18[ieta]);
    eHOEta17_[ieta]->Fill(eHOE17[ieta]);
    eHOEta18_[ieta]->Fill(eHOE18[ieta]);
    nHOE1_[ieta]->Fill((double)(nHO));
    nHOE2_[ieta]->Fill(phiInc,(double)(nHO));
    nHOE1T_[ieta]->Fill((double)(nHOT));
    nHOE2T_[ieta]->Fill(phiInc,(double)(nHOT));
    int nHOE=0, nHOET=0;
    if (eHOE17[ieta] > 0) nHOE++; if (eHOE17T[ieta] > 0) nHOET++;
    if (eHOE18[ieta] > 0) nHOE++; if (eHOE18T[ieta] > 0) nHOET++;
    nHOEta1_[ieta]->Fill((double)(nHOE));
    nHOEta2_[ieta]->Fill(phiInc,(double)(nHOE));
    nHOEta1T_[ieta]->Fill((double)(nHOET));
    nHOEta2T_[ieta]->Fill(phiInc,(double)(nHOET));
  }
  
  LogDebug("HitStudy") << "HOSimHitStudy::analyzeHits: Hits in EB " << nhit[0]
		       << " in " << ebEtow.size() << " towers with total E "
		       << etot[0] << "|" << etotT[0]
		       <<"\n                            Hits in HB "
		       << nhit[1] << " in " << hbEtow.size()
		       << " towers with total E " << etot[1] << "|" << etotT[1]
		       << "\n                            Hits in HO "
		       << nhit[2] << " in " << hoEtow.size()
		       << " towers with total E " << etot[2] << "|" << etotT[2]
		       << "\n                            Energy in EB " << eEB
		       << "|" << eEBT << " with HB " << eEBHB << "|" << eEBHBT
		       << " and with HO " << eEBHBHO << "|" << eEBHBHOT
		       << "\n                            E in HO layers "
		       << eHO17 <<"|" << eHO17T <<" " << eHO18 <<"|" << eHO18T
		       << " number of HO hits " << nHO << "|" << nHOT;

  if (eEBHBHO > 0.75*maxEnergy && print_) {
    edm::LogInfo("HitStudy") << "Event with excessive energy: EB = " << eEB
			     << " EB+HB = " << eEBHB << " EB+HB+HO = " 
			     << eEBHBHO;
    const std::string Dets[3] = {"EB", "HB", "HO"};
    for (int k=0; k<2; k++) {
      int nHit;
      if (k == 0) {
	nHit = ecalHits.size();
      } else {
	nHit = hcalHits.size();
      }
      for (int i=0; i<nHit; i++) {
	double       edep, time;
	int          indx = -1;
	unsigned int id_;
	int          ieta, iphi, depth=0;
	if (k == 0) {
	  indx             = 0;
	  edep             = ecalHits[i].energy();
	  time             = ecalHits[i].time();
	  id_              = ecalHits[i].id();
	  EBDetId id       = EBDetId(id_);
	  ieta             = id.ieta();
	  iphi             = id.iphi();
	} else {
	  indx             = -1;
	  edep             = hcalHits[i].energy();
	  time             = hcalHits[i].time();
	  id_              = hcalHits[i].id();
	  HcalDetId id     = HcalDetId(id_);
	  int subdet       = id.subdet();
	  if      (subdet == static_cast<int>(HcalBarrel))  indx = 1;
	  else if (subdet == static_cast<int>(HcalOuter))   indx = 2;
	  ieta             = id.ieta();
	  iphi             = id.iphi();
	  depth            = id.depth();
	}
	if (indx >= 0) {
	  edm::LogInfo("HitStudy") << Dets[indx] << " " << i << std::hex  <<id_
				   << std::dec << " (" << ieta << "|" << iphi 
				   << "|" << depth << ") " << std::setw(8)
				   << edep << " " << std::setw(8) << time;
	}
      }
    }
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(HOSimHitStudy);

#include "SimG4CMS/Calo/test/HOSimHitStudy.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "SimG4CMS/Calo/interface/CaloHitID.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "CLHEP/Units/PhysicalConstants.h"
#include <iostream>
#include <iomanip>

HOSimHitStudy::HOSimHitStudy(const edm::ParameterSet& ps) {

  g4Label   = ps.getUntrackedParameter<std::string>("moduleLabel","g4SimHits");
  hitLab[0] = ps.getUntrackedParameter<std::string>("EBCollection","EcalHitsEB");
  hitLab[1] = ps.getUntrackedParameter<std::string>("HCCollection","HcalHits");
  maxEnergy = ps.getUntrackedParameter<double>("MaxEnergy", 200.0);
  scaleEB   = ps.getUntrackedParameter<double>("ScaleEB", 1.0);
  scaleHB   = ps.getUntrackedParameter<double>("ScaleHB", 100.0);
  scaleHO   = ps.getUntrackedParameter<double>("ScaleHO", 2.0);
  edm::LogInfo("HitStudy") << "Module Label: " << g4Label << "   Hits: "
			   << hitLab[0] << ", " << hitLab[1] 
			   << "   MaxEnergy: " << maxEnergy
			   << "   Scale factor for EB " << scaleEB 
			   << ", for HB " <<scaleHB <<" and for HO " <<scaleHO;

  edm::Service<TFileService> tfile;
 
  if ( !tfile.isAvailable() )
    throw cms::Exception("BadConfig") << "TFileService unavailable: "
                                      << "please add it to config file";
  std::string dets[3] = {"EB", "HB", "HO"};
  char  name[20], title[60];
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
    sprintf (name, "EdepZone%d", i);
    sprintf (title, "Energy deposit (GeV) in %s", dets[i].c_str());
    edepZon_[i]  = tfile->make<TH1F>(name, title, 5000, 0., ymax);
    edepZon_[i]->GetXaxis()->SetTitle(title); 
    edepZon_[i]->GetYaxis()->SetTitle("Events");
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
}

void HOSimHitStudy::analyze(const edm::Event& e, const edm::EventSetup& ) {

  LogDebug("HitStudy") << "Run = " << e.id().run() << " Event = " 
		       << e.id().event();

  edm::Handle<edm::HepMCProduct > EvtHandle;
  e.getByLabel("source", EvtHandle);
  const  HepMC::GenEvent* myGenEvent = EvtHandle->GetEvent();

  eInc = etaInc = phiInc = 0;
  HepMC::GenEvent::particle_const_iterator p=myGenEvent->particles_begin();
  if (p != myGenEvent->particles_end()) {
    eInc   = (*p)->momentum().e();
    etaInc = (*p)->momentum().eta();
    phiInc = (*p)->momentum().phi();
  }

  for (int i=0; i<2; i++) {
    bool getHits = false;
    if (i == 0) ecalHits.clear();
    else        hcalHits.clear();
    edm::Handle<edm::PCaloHitContainer> hitsCalo;
    e.getByLabel(g4Label,hitLab[i],hitsCalo); 
    if (hitsCalo.isValid()) getHits = true;
    LogDebug("HitStudy") << "HcalValidation: Input flags Hits " << getHits;

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
  double etot[3];
  std::vector<unsigned int> ebID, hbID, hoID;
  std::vector<double>       ebEtow, hbEtow, hoEtow;
  for (int k=0; k<3; k++) {
    nhit[k] = 0;
    etot[k] = 0;
  }
  eneInc_->Fill(eInc);
  etaInc_->Fill(etaInc);
  phiInc_->Fill(phiInc);
  
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
	HcalDetId id     = HcalDetId(id_);
	int subdet       = id.subdet();
	if      (subdet == static_cast<int>(HcalBarrel))  indx = 1;
	else if (subdet == static_cast<int>(HcalOuter))   indx = 2;
      }
      if (indx >= 0) {
	time_[indx]->Fill(time);
	edep_[indx]->Fill(edep);
	etot[indx] += edep;
	nhit[indx]++;
	if (indx == 0) {
	  bool ok = false;
	  for (unsigned int j=0; j<ebID.size(); j++) {
	    if (id_ == ebID[j]) {
	      ebEtow[j] += edep;
	      ok         = true;
	      break;
	    }
	  }
	  if (!ok) {
	    ebID.push_back(id_);
	    ebEtow.push_back(edep);
	  }
	} else if (indx == 1) {
	  bool ok = false;
	  for (unsigned int j=0; j<hbID.size(); j++) {
	    if (id_ == hbID[j]) {
	      hbEtow[j] += edep;
	      ok         = true;
	      break;
	    }
	  }
	  if (!ok) {
	    hbID.push_back(id_);
	    hbEtow.push_back(edep);
	  }
	} else {
	  bool ok = false;
	  for (unsigned int j=0; j<hoID.size(); j++) {
	    if (id_ == hoID[j]) {
	      hoEtow[j] += edep;
	      ok         = true;
	      break;
	    }
	  }
	  if (!ok) {
	    hoID.push_back(id_);
	    hoEtow.push_back(edep);
	  }
	}
      }
    }
  }

  // Now for towers and total energy deposits
  for (int k=0; k<3; k++) {
    hit_[k]->Fill(double(nhit[k]));
    edepZon_[k]->Fill(etot[k]);
  }
  hitTow_[0]->Fill(double(ebEtow.size()));
  for (unsigned int i=0; i<ebEtow.size(); i++) edepTW_[0]->Fill(ebEtow[i]);
  hitTow_[1]->Fill(double(hbEtow.size()));
  for (unsigned int i=0; i<hbEtow.size(); i++) edepTW_[1]->Fill(hbEtow[i]);
  hitTow_[2]->Fill(double(hoEtow.size()));
  for (unsigned int i=0; i<hoEtow.size(); i++) edepTW_[2]->Fill(hoEtow[i]);
  double eEB     = scaleEB*etot[0];
  double eEBHB   = eEB + scaleHB*etot[1];
  double eEBHBHO = eEBHB + scaleHB*scaleHO*etot[2];
  eEB_->Fill(eEB);
  eEBHB_->Fill(eEBHB);
  eEBHBHO_->Fill(eEBHBHO);
  
  LogDebug("HitStudy") << "HOSimHitStudy::analyzeHits: Hits in EB " << nhit[0]
		       << " in " << ebEtow.size() << " towers with total E "
		       << etot[0] <<"\n                            Hits in HB "
		       << nhit[1] << " in " << hbEtow.size()
		       << " towers with total E " << etot[1]
		       << "\n                            Hits in HO "
		       << nhit[2] << " in " << hoEtow.size()
		       << " towers with total E " << etot[2]
		       << "\n                            Energy in EB " << eEB
		       << " with HB " << eEBHB << " and with HO " << eEBHBHO;

  if (eEBHBHO > 0.75*maxEnergy) {
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

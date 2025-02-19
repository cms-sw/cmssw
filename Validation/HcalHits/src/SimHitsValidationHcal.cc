#include "Validation/HcalHits/interface/SimHitsValidationHcal.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

//#include "FWCore/Utilities/interface/Exception.h"

SimHitsValidationHcal::SimHitsValidationHcal(const edm::ParameterSet& ps) {

  g4Label  = ps.getUntrackedParameter<std::string>("moduleLabel","g4SimHits");
  hcalHits = ps.getUntrackedParameter<std::string>("HitCollection","HcalHits");
  verbose_ = ps.getUntrackedParameter<bool>("Verbose", false);

  edm::LogInfo("HitsValidationHcal") << "Module Label: " << g4Label << "   Hits: "
				     << hcalHits;

  dbe_ = edm::Service<DQMStore>().operator->();
  if (dbe_) {
    if (verbose_) {
      dbe_->setVerbose(1);
      sleep (3);
      dbe_->showDirStructure();
    } else {
      dbe_->setVerbose(0);
    }
  }
}

SimHitsValidationHcal::~SimHitsValidationHcal() {}

void SimHitsValidationHcal::beginJob() {

  if (dbe_) {
    std::cout << "Histograms booked\n";
    dbe_->setCurrentFolder("HcalHitsV/SimHitsValidationHcal");

    //Histograms for Hits
    std::string divisions[25]={"HB0","HB1","HE0+z","HE1+z","HE2+z","HE0-z","HE1-z",
			       "HE2-z","HO0","HFL0+z","HFS0+z","HFL1+z","HFS1+z",
			       "HFL2+z","HFS2+z","HFL3+z","HFS3+z","HFL0-z","HFS0-z",
			       "HFL1-z","HFS1-z","HFL2-z","HFS2-z","HFL3-z","HFS3-z"};
    double etaLow[25]={-16,-16,16,16,16,-29,-29,-29,-15,29,29,29,29,29,29,29,29,
		       -41,-41,-41,-41,-41,-41,-41,-41};
    double etaHigh[25]={16,16,30,30,30,-30,-30,-30,15,41,41,41,41,41,41,41,41,
			-29,-29,-29,-29,-29,-29,-29,-29};
    int etaBins[25]={32,32,14,14,14,14,14,14,30,12,12,12,12,12,12,12,12,
		     12,12,12,12,12,12,12,12};
    char name[40], title[100];
    for (int i=0; i<25; ++i) {
      sprintf (name, "HcalHitEta%s", divisions[i].c_str());
      sprintf (name, "Hit energy as a function of eta tower index in %s", divisions[i].c_str());
      meHcalHitEta_[i] = dbe_->book1D(name, title, etaBins[i], etaLow[i], etaHigh[i]);
    }
  }
}

void SimHitsValidationHcal::endJob() {}

void SimHitsValidationHcal::analyze(const edm::Event& e, const edm::EventSetup& ) {

  LogDebug("HitsValidationHcal") << "Run = " << e.id().run() << " Event = " 
				 << e.id().event();

  std::vector<PCaloHit>               caloHits;
  edm::Handle<edm::PCaloHitContainer> hitsHcal;

  bool getHits = false;
  e.getByLabel(g4Label,hcalHits,hitsHcal); 
  if (hitsHcal.isValid()) getHits = true;
  
  LogDebug("HitsValidationHcal") << "HcalValidation: Input flags Hits " << getHits;

  if (getHits) {
    caloHits.insert(caloHits.end(),hitsHcal->begin(),hitsHcal->end());
    LogDebug("HitsValidationHcal") << "HcalValidation: Hit buffer " 
				   << caloHits.size(); 
    analyzeHits (caloHits);
  }
}

void SimHitsValidationHcal::analyzeHits (std::vector<PCaloHit>& hits) {

  int nHit = hits.size();
  double entotHB = 0, entotHE = 0, entotHF = 0, entotHO = 0; 
  int    nHB=0, nHE=0, nHO=0, nHF=0;

  for (int i=0; i<nHit; i++) {
    double energy    = hits[i].energy();
    double time      = hits[i].time();
    unsigned int id_ = hits[i].id();
    HcalDetId id     = HcalDetId(id_);
    int det          = id.det();
    int subdet       = id.subdet();
    int depth        = id.depth();
    int eta          = id.ieta();
    int phi          = id.iphi();
    unsigned int dep = hits[i].depth();
    int type         =-1;
    if (subdet == static_cast<int>(HcalBarrel)) {
      entotHB += energy;
      nHB++;
      type     = depth-1;
    } else if (subdet == static_cast<int>(HcalEndcap)) {
      entotHE += energy;
      nHE++;
      type     = depth+2;
      if (eta < 0) type += 3;
    } else if (subdet == static_cast<int>(HcalOuter)) {
      entotHO += energy;
      nHO++;
      type = 8;
    } else if (subdet == static_cast<int>(HcalForward)) {
      entotHF += energy;
      nHF++;
      type     = depth+8+2*dep;
      if (eta < 0) type += 8;
    }
    LogDebug("HitsValidationHcal") << "Hit[" << i << "] ID " << std::hex << id_ 
				   << std::dec << " Det " << det << " Sub " 
				   << subdet << " depth " << depth << " depthX "
				   << dep << " Eta " << eta << " Phi " << phi 
				   << " E " << energy << " time " << time
				   << " type " << type;
    double etax = eta - 0.5;
    if (eta < 0) etax += 1;
    if (dbe_ && type >= 0) {
      meHcalHitEta_[type]->Fill(etax,energy);
    }
  }

  /*
  if (dbe_) {
    if( entotHB != 0 ) for( int i=0; i<140; i++ ) meHBL10EneP_->Fill( -10.+(float(i)+0.5)/10., encontHB[i]/entotHB );
    if( entotHE != 0 ) for( int i=0; i<140; i++ ) meHEL10EneP_->Fill( -10.+(float(i)+0.5)/10., encontHE[i]/entotHE );
    if( entotHF != 0 ) for( int i=0; i<140; i++ ) meHFL10EneP_->Fill( -10.+(float(i)+0.5)/10., encontHF[i]/entotHF );
    if( entotHO != 0 ) for( int i=0; i<140; i++ ) meHOL10EneP_->Fill( -10.+(float(i)+0.5)/10., encontHO[i]/entotHO );
    meAllNHit_->Fill(double(nHit));
    meHBNHit_->Fill(double(nHB));
    meHENHit_->Fill(double(nHE));
    meHONHit_->Fill(double(nHO));
    meHFNHit_->Fill(double(nHF));
  }
  */
  LogDebug("HitsValidationHcal") << "SimHitsValidationHcal::analyzeHits: HB " << nHB 
				 << " HE " << nHE << " HO " << nHO << " HF " << nHF 
				 << " All " << nHit;

}


#include "Validation/HcalHits/interface/SimHitsValidationHcal.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

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
    edm::LogInfo("HitsValidationHcal") << "Histograms booked";
    dbe_->setCurrentFolder("HcalHitsV/SimHitsValidationHcal");

    //Histograms for Hits
    
    std::string divisions[nType]={"HB0","HB1","HE0+z","HE1+z","HE2+z","HE0-z","HE1-z",
				  "HE2-z","HO0","HFL0+z","HFS0+z","HFL1+z","HFS1+z",
				  "HFL2+z","HFS2+z","HFL3+z","HFS3+z","HFL0-z","HFS0-z",
				  "HFL1-z","HFS1-z","HFL2-z","HFS2-z","HFL3-z","HFS3-z"};
    
    std::string divisions1[nType]={"HB depth1","HB depth2 ","HE+z depth1","HE+z depth2","HE+z depth3","HE-z depth1","HE-z depth2",
				   "HE-z depth3","HO depth1","HFL+z depth1","HFS+z depth1","HFL+z depth2","HFS+z depth2",
				   "HFL+z depth3","HFS+z depth3","HFL+z depth4","HFS+z depth4","HFL-z depth1","HFS-z depth1",
				   "HFL-z depth2","HFS-z depth2","HFL-z depth3","HFS-z depth3 ","HFL-z depth4","HFS-z depth4"};
    
    double etaLow[nType]={-16,-16,16,16,16,-30,-30,-30,-15,29,29,29,29,29,29,29,29,
			  -41,-41,-41,-41,-41,-41,-41,-41};
    double etaHigh[nType]={16,16,30,30,30,-16,-16,-16,15,41,41,41,41,41,41,41,41,
			   -29,-29,-29,-29,-29,-29,-29,-29};
    int etaBins[nType]={32,32,14,14,14,14,14,14,30,12,12,12,12,12,12,12,12,
			12,12,12,12,12,12,12,12};
    char name[40], title[100];
    
    for (int i=0; i<nType; ++i) {
      sprintf (name, "HcalHitEta%s", divisions[i].c_str());
      sprintf (title, "Hit energy as a function of eta tower index in %s", divisions1[i].c_str());
      meHcalHitEta_[i] = dbe_->book1D(name, title, etaBins[i], etaLow[i], etaHigh[i]);
      
      sprintf (name, "HcalHitTimeAEta%s", divisions[i].c_str());
      sprintf (title, "Hit time as a function of eta tower index in %s", divisions1[i].c_str());
      meHcalHitTimeEta_[i] = dbe_->book1D(name, title, etaBins[i], etaLow[i], etaHigh[i]);
      
      sprintf (name, "HcalHitE25%s", divisions[i].c_str());
      sprintf (title, "Energy in time window 0 to 25 for a tower in %s", divisions1[i].c_str());
      meHcalEnergyl25_[i] = dbe_->book2D(name, title, etaBins[i], etaLow[i], etaHigh[i], 72, 0., 72.);
      
      sprintf (name, "HcalHitE50%s", divisions[i].c_str());
      sprintf (title, "Energy in time window 0 to 50 for a tower in %s", divisions1[i].c_str());
      meHcalEnergyl50_[i] = dbe_->book2D(name, title, etaBins[i], etaLow[i], etaHigh[i], 72, 0., 72.);
      
      sprintf (name, "HalHitE100%s", divisions[i].c_str());
      sprintf (title, "Energy in time window 0 to 100 for a tower in %s", divisions1[i].c_str());
      meHcalEnergyl100_[i] = dbe_->book2D(name, title, etaBins[i], etaLow[i], etaHigh[i], 72, 0., 72.);
      
      sprintf (name, "HcalHitE250%s", divisions[i].c_str());
      sprintf (title, "Energy in time window 0 to 250 for a tower in %s", divisions1[i].c_str());
      meHcalEnergyl250_[i] = dbe_->book2D(name, title, etaBins[i], etaLow[i], etaHigh[i], 72, 0., 72.);
    }

    sprintf (name, "Energy_HB");
    meEnergy_HB = dbe_->book1D(name, name, 100,0,1);
    sprintf (name, "Energy_HE");
    meEnergy_HE = dbe_->book1D(name, name, 100,0,1);
    sprintf (name, "Energy_HO");
    meEnergy_HO = dbe_->book1D(name, name, 100,0,1);
    sprintf (name, "Energy_HF");
    meEnergy_HF = dbe_->book1D(name, name, 100,0,50);
    
    sprintf (name, "Time_HB");
    metime_HB = dbe_->book1D(name, name, 300,-150,150);
    sprintf (name, "Time_HE");
    metime_HE = dbe_->book1D(name, name, 300,-150,150);
    sprintf (name, "Time_HO");
    metime_HO = dbe_->book1D(name, name, 300,-150, 150);
    sprintf (name, "Time_HF");
    metime_HF = dbe_->book1D(name, name, 300,-150,150);

    sprintf (name, "Time_Enweighted_HB");
    metime_enweighted_HB = dbe_->book1D(name, name, 300,-150,150);
    sprintf (name, "Time_Enweighted_HE");
    metime_enweighted_HE = dbe_->book1D(name, name, 300,-150,150);
    sprintf (name, "Time_Enweighted_HO");
    metime_enweighted_HO = dbe_->book1D(name, name, 300,-150, 150);
    sprintf (name, "Time_Enweighted_HF");
    metime_enweighted_HF = dbe_->book1D(name, name, 300,-150,150);
  }
}

void SimHitsValidationHcal::endJob() {}

void SimHitsValidationHcal::analyze(const edm::Event& e, const edm::EventSetup& ) {
  
  
  LogDebug("SimHitsValidationHcal") << "Run = " << e.id().run() << " Event = " 
				    << e.id().event();
  
  std::vector<PCaloHit>               caloHits;
  edm::Handle<edm::PCaloHitContainer> hitsHcal;
  
  bool getHits = false;
  e.getByLabel(g4Label,hcalHits,hitsHcal); 
  if (hitsHcal.isValid()) getHits = true;
  
  LogDebug("SimHitsValidationHcal") << "SimHitsValidationHcal.: Input flags Hits " << getHits;

  if (getHits) {
    caloHits.insert(caloHits.end(),hitsHcal->begin(),hitsHcal->end());
    LogDebug("SimHitsValidationHcal") << "SimHitsValidationHcal: Hit buffer " 
				      << caloHits.size(); 
    analyzeHits (caloHits);
  }
}

void SimHitsValidationHcal::analyzeHits (std::vector<PCaloHit>& hits) {

  int nHit = hits.size();
  double entotHB = 0, entotHE = 0, entotHF = 0, entotHO = 0; 
  double timetotHB = 0, timetotHE = 0, timetotHF = 0, timetotHO = 0; 
  int    nHB=0, nHE=0, nHO=0, nHF=0;
  
  std::map<std::pair<HcalDetId,int>,energysum> map_try;
  map_try.clear();
  
  std::map<std::pair<HcalDetId,int>,energysum>::iterator itr;
  
  for (int i=0; i<nHit; i++) {
    double energy    = hits[i].energy();
    double time      = hits[i].time();
    unsigned int id_ = hits[i].id();
    HcalDetId id     = HcalDetId(id_);
    int itime        = (int)(time);
    int det          = id.det();
    int subdet       = id.subdet();
    int depth        = id.depth();
    int eta          = id.ieta();
    int phi          = id.iphi();
    unsigned int dep = hits[i].depth();
        
    int type         =-1;
    if (subdet == static_cast<int>(HcalBarrel)) {
      entotHB += energy;
      timetotHB += time;
      nHB++;
      type     = depth-1;
    } else if (subdet == static_cast<int>(HcalEndcap)) {
      entotHE += energy;
      timetotHE += time;
      nHE++;
      type     = depth+1;
      if (eta < 0) type += 3;
    } else if (subdet == static_cast<int>(HcalOuter)) {
      entotHO += energy;
      timetotHO += time;
      nHO++;
      type = 8;
    } else if (subdet == static_cast<int>(HcalForward)) {
      entotHF += energy;
      timetotHF += time;
      nHF++;
      type     = depth+8+2*dep;
      if (eta < 0) type += 8;
    }

    std::pair<HcalDetId,int> id0(id,type);
    //   LogDebug("HitsValidationHcal") << "Id and type " << id << ":" << type;
    energysum ensum;
    if (map_try.count(id0) != 0) ensum =  map_try[id0];
    if (itime<250) {
      ensum.e250 += energy;
      if (itime<100) {
	ensum.e100 += energy;
	if (itime<50) {
	  ensum.e50 += energy;
	  if (itime<25) ensum.e25 += energy;
	}
      }
    }
    map_try[id0] = ensum;

    LogDebug("SimHitsValidationHcal") << "Hit[" << i << "] ID " << std::hex << id_ 
				      << std::dec << " " << id << std::dec<< " Det " << det << " Sub " 
				      << subdet << " depth " << depth << " depthX "
				      << dep << " Eta " << eta << " Phi " << phi 
				      << " E " << energy << " time " << time
				      << " type " << type;

    //    LogDebug("HitsValidationHcal") << "Hit[" << i << "] ID " << std::hex << id_ << "ID="<<std::dec << id << " Det " << det << " Sub " << subdet << " depth " << depth << " depthX " << dep << " Eta " << eta << " Phi " << phi << " E  " << energy << " time  " << time <<" itime  " << itime << " type " << type;
    
    double etax = eta - 0.5;
    if (eta < 0) etax += 1;
    if (dbe_ && type >= 0) {
      meHcalHitEta_[type]->Fill(etax,energy);
      meHcalHitTimeEta_[type]->Fill(etax,time);
    }
  }
  
  if (dbe_) {
    meEnergy_HB->Fill(entotHB);
    meEnergy_HE->Fill(entotHE);
    meEnergy_HF->Fill(entotHF);
    meEnergy_HO->Fill(entotHO);

    metime_HB->Fill(timetotHB);
    metime_HE->Fill(timetotHE);
    metime_HF->Fill(timetotHF);
    metime_HO->Fill(timetotHO);
    
    metime_enweighted_HB->Fill(timetotHB,entotHB);
    metime_enweighted_HE->Fill(timetotHE,entotHE);
    metime_enweighted_HF->Fill(timetotHF,entotHF);
    metime_enweighted_HO->Fill(timetotHO,entotHO);
  }
  
  for ( itr = map_try.begin() ; itr != map_try.end(); ++itr)   {
    if (dbe_ && (*itr).first.second >= 0) {
      HcalDetId id= (*itr).first.first;
      energysum ensum= (*itr).second;
      int eta = id.ieta();
      int phi = id.iphi();
      double etax= eta-0.5;
      double phix= phi-0.5;
      meHcalEnergyl25_[(*itr).first.second]->Fill(etax,phix,ensum.e25);
      meHcalEnergyl50_[(*itr).first.second]->Fill(etax,phix,ensum.e50);
      meHcalEnergyl100_[(*itr).first.second]->Fill(etax,phix,ensum.e100);
      meHcalEnergyl250_[(*itr).first.second]->Fill(etax,phix,ensum.e250);

      //LogDebug("HitsValidationHcal") <<" energy of tower ="<<(*itr).first<<"in time 25ns is== "<<(*itr).second.e25<<"in time 25-50ns=="<<(*itr).second.e50<<"in time 50-100ns=="<<(*itr).second.e100<<"in time 100-250 ns=="<<(*itr).second.e250;
    }
  }
  
}

DEFINE_FWK_MODULE(SimHitsValidationHcal);

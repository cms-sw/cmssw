#include "Validation/HcalHits/interface/SimHitsValidationHcal.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"
#include "SimDataFormats/CaloTest/interface/HcalTestNumbering.h"

//#define DebugLog

SimHitsValidationHcal::SimHitsValidationHcal(const edm::ParameterSet& ps) :
  initialized(false) {

  g4Label_    = ps.getUntrackedParameter<std::string>("moduleLabel","g4SimHits");
  hcalHits_   = ps.getUntrackedParameter<std::string>("HitCollection","HcalHits");
  verbose_    = ps.getUntrackedParameter<bool>("Verbose", false);
  testNumber_ = ps.getUntrackedParameter<bool>("TestNumber", false);

  edm::LogInfo("SimHitsValidationHcal") << "Module Label: " << g4Label_ 
					<< " Hits: " << hcalHits_ 
					<< " TestNumbering " << testNumber_;

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

SimHitsValidationHcal::~SimHitsValidationHcal() { }

void SimHitsValidationHcal::beginRun(const edm::Run&, 
				     const edm::EventSetup& iSetup) {

  if (!(initialized)) {
    edm::ESHandle<HcalDDDRecConstants> pHRNDC;
    iSetup.get<HcalRecNumberingRecord>().get( pHRNDC );
    hcons = &(*pHRNDC);
    initialized = true;
    maxDepthHB_ = hcons->getMaxDepth(0);
    maxDepthHE_ = hcons->getMaxDepth(1);
    maxDepthHF_ = hcons->getMaxDepth(2);
    maxDepthHO_ = hcons->getMaxDepth(3);
#ifdef DebugLog
    std::cout << " Maximum Depths HB:"<< maxDepthHB_ << " HE:" << maxDepthHE_ 
	      << " HO:" << maxDepthHO_ << " HF:" << maxDepthHF_ << std::endl;
#endif  
    std::vector<std::pair<std::string,std::string> > divisions = getHistogramTypes();
    if (dbe_) {
      edm::LogInfo("SimHitsValidationHcal") << "Booking the Histograms";
      dbe_->setCurrentFolder("HcalHitsV/SimHitsValidationHcal");

      char name[100], title[200];
      for (unsigned int i=0; i<types.size(); ++i) {
	etaRange limit = getLimits(types[i]);
	sprintf (name, "HcalHitEta%s", divisions[i].first.c_str());
	sprintf (title, "Hit energy as a function of eta tower index in %s", divisions[i].second.c_str());
	meHcalHitEta_.push_back(dbe_->book1D(name, title, limit.bins, limit.low, limit.high));
      
	sprintf (name, "HcalHitTimeAEta%s", divisions[i].first.c_str());
	sprintf (title, "Hit time as a function of eta tower index in %s", divisions[i].second.c_str());
	meHcalHitTimeEta_.push_back(dbe_->book1D(name, title, limit.bins, limit.low, limit.high));
      
	sprintf (name, "HcalHitE25%s", divisions[i].first.c_str());
	sprintf (title, "Energy in time window 0 to 25 for a tower in %s", divisions[i].second.c_str());
	meHcalEnergyl25_.push_back(dbe_->book2D(name, title, limit.bins, limit.low, limit.high, 72, 0., 72.));
      
	sprintf (name, "HcalHitE50%s", divisions[i].first.c_str());
	sprintf (title, "Energy in time window 0 to 50 for a tower in %s", divisions[i].second.c_str());
	meHcalEnergyl50_.push_back(dbe_->book2D(name, title, limit.bins, limit.low, limit.high, 72, 0., 72.));
      
	sprintf (name, "HcalHitE100%s", divisions[i].first.c_str());
	sprintf (title, "Energy in time window 0 to 100 for a tower in %s", divisions[i].second.c_str());
	meHcalEnergyl100_.push_back(dbe_->book2D(name, title, limit.bins, limit.low, limit.high, 72, 0., 72.));
      
	sprintf (name, "HcalHitE250%s", divisions[i].first.c_str());
	sprintf (title, "Energy in time window 0 to 250 for a tower in %s", divisions[i].second.c_str());
	meHcalEnergyl250_.push_back(dbe_->book2D(name, title, limit.bins, limit.low, limit.high, 72, 0., 72.));
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
}

void SimHitsValidationHcal::analyze(const edm::Event& e, 
				    const edm::EventSetup&) {
#ifdef DebugLog  
  std::cout << "Run = " << e.id().run() << " Event = " << e.id().event()
	    << std::endl;
#endif
  std::vector<PCaloHit>               caloHits;
  edm::Handle<edm::PCaloHitContainer> hitsHcal;
  
  bool getHits = false;
  e.getByLabel(g4Label_,hcalHits_,hitsHcal); 
  if (hitsHcal.isValid()) getHits = true;
#ifdef DebugLog  
  std::cout << "SimHitsValidationHcal.: Input flags Hits " << getHits 
	    << std::endl;
#endif
  if (getHits) {
    caloHits.insert(caloHits.end(),hitsHcal->begin(),hitsHcal->end());
#ifdef DebugLog
    std::cout << "testNumber_:" << testNumber_ << std::endl;
#endif
    if (testNumber_) {
      for (unsigned int i=0; i<caloHits.size(); ++i) {
	unsigned int id_ = caloHits[i].id();
	int subdet, z, depth0, eta0, phi0, lay;
	HcalTestNumbering::unpackHcalIndex(id_, subdet, z, depth0, eta0, phi0, lay);
	int sign = (z==0) ? (-1):(1);
#ifdef DebugLog
	std::cout << "Hit[" << i << "] subdet|z|depth|eta|phi|lay " << subdet 
		  << "|" << z << "|" << depth0 << "|" << eta0 << "|" << phi0 
		  << "|" << lay << std::endl;
#endif
	HcalDDDRecConstants::HcalID id = hcons->getHCID(subdet, eta0, phi0, lay, depth0);
	
	HcalDetId hid;
	if (subdet==int(HcalBarrel)) {
	  hid = HcalDetId(HcalBarrel,sign*id.eta,id.phi,id.depth);        
	} else if (subdet==int(HcalEndcap)) {
	  hid = HcalDetId(HcalEndcap,sign*id.eta,id.phi,id.depth);    
	} else if (subdet==int(HcalOuter)) {
	  hid = HcalDetId(HcalOuter,sign*id.eta,id.phi,id.depth);    
	} else if (subdet==int(HcalForward)) {
	  hid = HcalDetId(HcalForward,sign*id.eta,id.phi,id.depth);
	}
	caloHits[i].setID(hid.rawId());
#ifdef DebugLog
	std::cout << "Hit[" << i << "] " << hid << std::endl;
#endif
      }
    }
#ifdef DebugLog
    std::cout << "SimHitsValidationHcal: Hit buffer " << caloHits.size()
	      << std::endl; 
#endif
    analyzeHits (caloHits);
  }
}

void SimHitsValidationHcal::analyzeHits (std::vector<PCaloHit>& hits) {

  int    nHit = hits.size();
  double entotHB = 0, entotHE = 0, entotHF = 0, entotHO = 0; 
  double timetotHB = 0, timetotHE = 0, timetotHF = 0, timetotHO = 0; 
  int    nHB=0, nHE=0, nHO=0, nHF=0;
  
  std::map<std::pair<HcalDetId,unsigned int>,energysum> map_try;
  map_try.clear();
  
  std::map<std::pair<HcalDetId,unsigned int>,energysum>::iterator itr;
  
  for (int i=0; i<nHit; i++) {
    double energy    = hits[i].energy();
    double time      = hits[i].time();
    HcalDetId id     = HcalDetId(hits[i].id());
    int itime        = (int)(time);
    int subdet       = id.subdet();
    int depth        = id.depth();
    int eta          = id.ieta();
    unsigned int dep = hits[i].depth();
        
    std::pair<int,int> types = histId(subdet, eta, depth, dep);
    if (subdet == static_cast<int>(HcalBarrel)) {
      entotHB   += energy;
      timetotHB += time;
      nHB++;
    } else if (subdet == static_cast<int>(HcalEndcap)) {
      entotHE += energy;
      timetotHE += time;
      nHE++;
    } else if (subdet == static_cast<int>(HcalOuter)) {
      entotHO += energy;
      timetotHO += time;
      nHO++;
    } else if (subdet == static_cast<int>(HcalForward)) {
      entotHF += energy;
      timetotHF += time;
      nHF++;
    }

    std::pair<HcalDetId,unsigned int> id0(id,dep);
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
    
#ifdef DebugLog
    std::cout << "Hit[" << i << "] ID " << std::dec << " " << id << std::dec
	      << " Det " << id.det() << " Sub " << subdet << " depth " << depth
	      << " depthX "<< dep << " Eta " << eta << " Phi " << id.iphi()
	      << " E " << energy << " time " << time
	      << " type " << types.first << " " << types.second
	      << std::endl;
#endif
    double etax = eta - 0.5;
    if (eta < 0) etax += 1;
    if (dbe_) {
      if (types.first >= 0) {
	meHcalHitEta_[types.first]->Fill(etax,energy);
	meHcalHitTimeEta_[types.first]->Fill(etax,time);
      }
      if (types.second >= 0) {
	meHcalHitEta_[types.second]->Fill(etax,energy);
	meHcalHitTimeEta_[types.second]->Fill(etax,time);
      }
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
  
  for (itr = map_try.begin() ; itr != map_try.end(); ++itr)   {
    if (dbe_) {
      HcalDetId id    = (*itr).first.first;
      energysum ensum = (*itr).second;
      std::pair<int,int> types = histId((int)(id.subdet()), id.ieta(), id.depth(), (*itr).first.second);
      int eta = id.ieta();
      int phi = id.iphi();
      double etax= eta-0.5;
      double phix= phi-0.5;
      if (types.first >= 0) {
	meHcalEnergyl25_[types.first]->Fill(etax,phix,ensum.e25);
	meHcalEnergyl50_[types.first]->Fill(etax,phix,ensum.e50);
	meHcalEnergyl100_[types.first]->Fill(etax,phix,ensum.e100);
	meHcalEnergyl250_[types.first]->Fill(etax,phix,ensum.e250);
      }
      if (types.second >= 0) {
	meHcalEnergyl25_[types.second]->Fill(etax,phix,ensum.e25);
	meHcalEnergyl50_[types.second]->Fill(etax,phix,ensum.e50);
	meHcalEnergyl100_[types.second]->Fill(etax,phix,ensum.e100);
	meHcalEnergyl250_[types.second]->Fill(etax,phix,ensum.e250);
      }
      
#ifdef DebugLog
      std::cout << " energy of tower =" << (*itr).first.first 
		<< " in time 25ns is == " << (*itr).second.e25 
		<< " in time 25-50ns == " << (*itr).second.e50 
		<< " in time 50-100ns == " << (*itr).second.e100 
		<< " in time 100-250 ns == " << (*itr).second.e250 
		<< std::endl;
#endif
    }
  }
  
}

SimHitsValidationHcal::etaRange SimHitsValidationHcal::getLimits (idType type){

  int    bins;
  std::pair<int,int> range;
  double low, high;
  
  if (type.subdet == HcalBarrel) {
    range = hcons->getEtaRange(0);
    low  =-range.second;
    high = range.second;
    bins = (high-low);
  } else if (type.subdet == HcalEndcap) {
    range = hcons->getEtaRange(1);
    bins = range.second- range.first;
    if (type.z == 1) {
      low  = range.first;
      high = range.second;
    } else {
      low  =-range.second;
      high =-range.first;
    }
  } else if (type.subdet == HcalOuter) {
    range = hcons->getEtaRange(3);
    low  =-range.second;
    high = range.second;
    bins = high-low;
  } else if (type.subdet == HcalForward) {
    range = hcons->getEtaRange(2);
    bins = range.second-range.first;
    if (type.z == 1) {
      low  = range.first;
      high = range.second;
    } else {
      low  =-range.second;
      high =-range.first;
    }
  } else {
    bins = 82;
    low  =-41;
    high = 41;
  }
#ifdef DebugLog
  std::cout << "Subdetector:" << type.subdet << " z:" << type.z 
	    << " range.first:" << range.first << " and second:" 
	    << range.second << std::endl;
  std::cout << "bins: " << bins << " low:" << low << " high:" << high
	    << std::endl;
#endif
  return SimHitsValidationHcal::etaRange(bins, low, high);
}

std::pair<int,int> SimHitsValidationHcal::histId(int subdet, int eta, int depth, unsigned int dep) {

  int id1(-1), id2(-1);
  for (unsigned int k=0; k<types.size(); ++k) {
    if (subdet == HcalForward) {
      if (subdet == (int)(types[k].subdet) && depth == types[k].depth1 &&
	  eta*types[k].z > 0 && dep == (unsigned int)(types[k].depth2)) {
	id1 = k; break;
      }
    } else if (subdet == HcalEndcap) {
      if (subdet == (int)(types[k].subdet) && depth == types[k].depth1 &&
	  eta*types[k].z > 0) {
	id1 = k; break;
      }
    } else {
      if (subdet == (int)(types[k].subdet) && depth == types[k].depth1) {
	id1 = k; break;
      }
    }
  }
  if (subdet == HcalForward) depth += 2*dep;
  for (unsigned int k=0; k<types.size(); ++k) {
    if (types[k].subdet == HcalEmpty && types[k].depth1 == depth) {
      id2 = k; 
      break;
    }
  }
  
  return std::pair<int,int>(id1,id2);
}

std::vector<std::pair<std::string,std::string> > SimHitsValidationHcal::getHistogramTypes() {

  int maxDepth = std::max(maxDepthHB_,maxDepthHE_);
  maxDepth = std::max(maxDepth,maxDepthHF_);
  maxDepth = std::max(maxDepth,maxDepthHO_);

  std::vector<std::pair<std::string,std::string> > divisions;
  std::pair<std::string,std::string> names;
  char                               name1[20], name2[20];
  SimHitsValidationHcal::idType      type;
  //first overall Hcal
  for (int depth=0; depth<maxDepth; ++depth) {
    sprintf (name1, "HC%d", depth);
    sprintf (name2, "HCAL depth%d", depth+1);
    names = std::pair<std::string,std::string>(std::string(name1),std::string(name2));
    type  = SimHitsValidationHcal::idType(HcalEmpty,0,depth+1,depth+1);
    divisions.push_back(names);
    types.push_back(type);
  }
  //HB
  for (int depth=0; depth<maxDepthHB_; ++depth) {
    sprintf (name1, "HB%d", depth);
    sprintf (name2, "HB depth%d", depth+1);
    names = std::pair<std::string,std::string>(std::string(name1),std::string(name2));
    type  = SimHitsValidationHcal::idType(HcalBarrel,0,depth+1,depth+1);
    divisions.push_back(names);
    types.push_back(type);
  }
  //HE
  for (int depth=0; depth<maxDepthHE_; ++depth) {
    sprintf (name1, "HE%d+z", depth);
    sprintf (name2, "HE +z depth%d", depth+1);
    names = std::pair<std::string,std::string>(std::string(name1),std::string(name2));
    type  = SimHitsValidationHcal::idType(HcalEndcap,1,depth+1,depth+1);
    divisions.push_back(names);
    types.push_back(type);
    sprintf (name1, "HE%d-z", depth);
    sprintf (name2, "HE -z depth%d", depth+1);
    names = std::pair<std::string,std::string>(std::string(name1),std::string(name2));
    type  = SimHitsValidationHcal::idType(HcalEndcap,-1,depth+1,depth+1);
    divisions.push_back(names);
    types.push_back(type);
  }
  //HO
  {
    int depth = maxDepthHO_;
    sprintf (name1, "HO%d", depth);
    sprintf (name2, "HO depth%d", depth);
    names = std::pair<std::string,std::string>(std::string(name1),std::string(name2));
    type  = SimHitsValidationHcal::idType(HcalOuter,0,depth,depth);
    divisions.push_back(names);
    types.push_back(type);
  }
  //HF (first absorber, then different types of abnormal hits)
  std::string hfty1[4] = {"A","W","B","J"};
  std::string hfty2[4] = {"Absorber", "Window", "Bundle", "Jungle"};
  int         dept0[4] = {0, 1, 2, 3};
  for (int k=0; k<4; ++k) {
    for (int depth=0; depth<maxDepthHF_; ++depth) {
      sprintf (name1, "HF%s%d+z", hfty1[k].c_str(), depth);
      sprintf (name2, "HF (%s) +z depth%d", hfty2[k].c_str(), depth+1);
      names = std::pair<std::string,std::string>(std::string(name1),std::string(name2));
      type  = SimHitsValidationHcal::idType(HcalForward,1,depth+1,dept0[k]);
      divisions.push_back(names);
      types.push_back(type);
      sprintf (name1, "HF%s%d-z", hfty1[k].c_str(), depth);
      sprintf (name2, "HF (%s) -z depth%d", hfty2[k].c_str(), depth+1);
      names = std::pair<std::string,std::string>(std::string(name1),std::string(name2));
      type  = SimHitsValidationHcal::idType(HcalForward,-1,depth+1,dept0[k]);
      divisions.push_back(names);
      types.push_back(type);
    }
  }

  return divisions;
    
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(SimHitsValidationHcal);

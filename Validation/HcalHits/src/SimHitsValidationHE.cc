#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "Geometry/HcalCommonData/interface/HcalDDDRecConstants.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"
#include "SimDataFormats/CaloTest/interface/HcalTestNumbering.h"

#define DebugLog

#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>

class SimHitsValidationHE: public edm::EDAnalyzer{
public:

  SimHitsValidationHE(const edm::ParameterSet& ps);
  ~SimHitsValidationHE();

protected:

  void beginRun(const edm::Run& r,  const edm::EventSetup& c);
  void beginJob () {}
  void endJob   () {}
  void analyze  (const edm::Event& e, const edm::EventSetup& c);

private:

  struct energysum {
    energysum() {e25=e50=e100=e250=0.0;}
    double e25, e50, e100, e250;
  };

  struct idType {
    idType() {z=depth1=depth2=0;}
    idType(int iz, int d1, int d2) {z=iz; depth1=d1; depth2=d2;}
    int             z, depth1, depth2;
  };

  struct etaRange {
    etaRange() {bins=0; low=high=0;}
    etaRange(int bin, double min, double max) {bins=bin; low=min; high=max;}
    int             bins;
    double          low, high;
  };

  std::vector<std::pair<std::string,std::string> > getHistogramTypes();
  void analyzeHits  (std::vector<PCaloHit> &);
  etaRange getLimits (idType);
  int histId(int eta, int depth, unsigned int dep);

  bool                initialized;
  std::string         g4Label_, hcalHits_;
  const HcalDDDRecConstants *hcons;
  std::vector<idType> types;
  bool                verbose_, testNumber_;
  int                 maxDepth_;
  DQMStore            *dbe_;


  std::vector<MonitorElement*> meHitEta_, meHitTimeEta_, meEnergyl25_;
  std::vector<MonitorElement*> meEnergyl50_, meEnergyl100_, meEnergyl250_;
  MonitorElement              *meEnergy_, *metime_, *metime_enweighted_;
};


SimHitsValidationHE::SimHitsValidationHE(const edm::ParameterSet& ps) :
  initialized(false) {

  g4Label_    = ps.getUntrackedParameter<std::string>("moduleLabel","g4SimHits");
  hcalHits_   = ps.getUntrackedParameter<std::string>("HitCollection","HcalHits");
  verbose_    = ps.getUntrackedParameter<bool>("Verbose", false);
  testNumber_ = ps.getUntrackedParameter<bool>("TestNumber", false);

  edm::LogInfo("SimHitsValidationHE") << "Module Label: " << g4Label_ 
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

SimHitsValidationHE::~SimHitsValidationHE() { }

void SimHitsValidationHE::beginRun(const edm::Run&, 
				   const edm::EventSetup& iSetup) {

  if (!(initialized)) {
    edm::ESHandle<HcalDDDRecConstants> pHRNDC;
    iSetup.get<HcalRecNumberingRecord>().get( pHRNDC );
    hcons = &(*pHRNDC);
    initialized = true;
    maxDepth_ = hcons->getMaxDepth(1);
#ifdef DebugLog
    std::cout << " Maximum Depths HE:"<< maxDepth_ << std::endl;
#endif  
    std::vector<std::pair<std::string,std::string> > divisions = getHistogramTypes();
    if (dbe_) {
      edm::LogInfo("SimHitsValidationHE") << "Booking the Histograms";
      dbe_->setCurrentFolder("HcalHitsV/SimHitsValidationHE");

      char name[100], title[200];
      for (unsigned int i=0; i<types.size(); ++i) {
	etaRange limit = getLimits(types[i]);
	sprintf (name, "HitEta%s", divisions[i].first.c_str());
	sprintf (title, "Hit energy as a function of eta tower index in %s", divisions[i].second.c_str());
	meHitEta_.push_back(dbe_->book1D(name, title, limit.bins, limit.low, limit.high));
      
	sprintf (name, "HitTimeEta%s", divisions[i].first.c_str());
	sprintf (title, "Hit time as a function of eta tower index in %s", divisions[i].second.c_str());
	meHitTimeEta_.push_back(dbe_->book1D(name, title, limit.bins, limit.low, limit.high));
      
	sprintf (name, "HitE25%s", divisions[i].first.c_str());
	sprintf (title, "Energy in time window 0 to 25 for a tower in %s", divisions[i].second.c_str());
	meEnergyl25_.push_back(dbe_->book2D(name, title, limit.bins, limit.low, limit.high, 72, 0., 72.));
      
	sprintf (name, "HitE50%s", divisions[i].first.c_str());
	sprintf (title, "Energy in time window 0 to 50 for a tower in %s", divisions[i].second.c_str());
	meEnergyl50_.push_back(dbe_->book2D(name, title, limit.bins, limit.low, limit.high, 72, 0., 72.));
      
	sprintf (name, "HitE100%s", divisions[i].first.c_str());
	sprintf (title, "Energy in time window 0 to 100 for a tower in %s", divisions[i].second.c_str());
	meEnergyl100_.push_back(dbe_->book2D(name, title, limit.bins, limit.low, limit.high, 72, 0., 72.));
      
	sprintf (name, "Hcal250%s", divisions[i].first.c_str());
	sprintf (title, "Energy in time window 0 to 250 for a tower in %s", divisions[i].second.c_str());
	meEnergyl250_.push_back(dbe_->book2D(name, title, limit.bins, limit.low, limit.high, 72, 0., 72.));
      }

      sprintf (name, "Energy_HE");
      meEnergy_ = dbe_->book1D(name, name, 100,0,1);
      sprintf (name, "Time_HE");
      metime_   = dbe_->book1D(name, name, 300,-150,150);
      sprintf (name, "Time_Enweighted_HE");
      metime_enweighted_ = dbe_->book1D(name, name, 300,-150,150);
    }
  }
}

void SimHitsValidationHE::analyze(const edm::Event& e, 
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
  std::cout << "SimHitsValidationHE.: Input flags Hits " << getHits 
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
	
	HcalDetId hid = ((subdet==int(HcalEndcap)) ? 
			 HcalDetId(HcalEndcap,sign*id.eta,id.phi,id.depth) :
			 HcalDetId(HcalEmpty,sign*id.eta,id.phi,id.depth));
	caloHits[i].setID(hid.rawId());
#ifdef DebugLog
	std::cout << "Hit[" << i << "] " << hid << std::endl;
#endif
      }
    }
#ifdef DebugLog
    std::cout << "SimHitsValidationHE: Hit buffer " << caloHits.size()
	      << std::endl; 
#endif
    analyzeHits (caloHits);
  }
}

void SimHitsValidationHE::analyzeHits (std::vector<PCaloHit>& hits) {

  int    nHit = hits.size();
  double entot = 0, timetot = 0;
  int    nHE=0;
  
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
        
    if (subdet == static_cast<int>(HcalEndcap)) {
      int types = histId(eta, depth, dep);
      entot   += energy;
      timetot += time;
      nHE++;

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
		<< " Det " << id.det() << " Sub " << subdet << " depth " 
		<< depth << " depthX "<< dep << " Eta " << eta << " Phi " 
		<< id.iphi() << " E " << energy << " time " << time
		<< " type " << types << std::endl;
#endif
      double etax = eta - 0.5;
      if (eta < 0) etax += 1;
      if (dbe_) {
	if (types >= 0) {
	  meHitEta_[types]->Fill(etax,energy);
	  meHitTimeEta_[types]->Fill(etax,time);
	}
      }
    }
  }
  
  if (dbe_) {
    meEnergy_->Fill(entot);
    metime_->Fill(timetot);
    metime_enweighted_->Fill(timetot,entot);
  }
  
  for (itr = map_try.begin() ; itr != map_try.end(); ++itr)   {
    if (dbe_) {
      HcalDetId id    = (*itr).first.first;
      energysum ensum = (*itr).second;
      int types  = histId(id.ieta(), id.depth(), (*itr).first.second);
      int eta    = id.ieta();
      int phi    = id.iphi();
      double etax= eta-0.5;
      double phix= phi-0.5;
      if (types >= 0) {
	meEnergyl25_[types]->Fill(etax,phix,ensum.e25);
	meEnergyl50_[types]->Fill(etax,phix,ensum.e50);
	meEnergyl100_[types]->Fill(etax,phix,ensum.e100);
	meEnergyl250_[types]->Fill(etax,phix,ensum.e250);
      }
      
#ifdef DebugLog
      std::cout << " energy of tower = "     << id
		<< " in time 25ns is == "    << ensum.e25 
		<< " in time 25-50ns == "    << ensum.e50 
		<< " in time 50-100ns == "   << ensum.e100 
		<< " in time 100-250 ns == " << ensum.e250 
		<< std::endl;
#endif
    }
  }
  
}

SimHitsValidationHE::etaRange SimHitsValidationHE::getLimits (idType type){

  int    bins;
  std::pair<int,int> range;
  double low, high;
  
  range = hcons->getEtaRange(1);
  bins = range.second- range.first;
  if (type.z == 1) {
    low  = range.first;
    high = range.second;
  } else {
    low  =-range.second;
    high =-range.first;
  }
#ifdef DebugLog
  std::cout << "z:" << type.z << " range.first:" << range.first 
	    << " and second:" << range.second << std::endl;
  std::cout << "bins: " << bins << " low:" << low << " high:" << high
	    << std::endl;
#endif
  return SimHitsValidationHE::etaRange(bins, low, high);
}

int SimHitsValidationHE::histId(int eta, int depth, unsigned int dep) {

  int id1(-1);
  for (unsigned int k=0; k<types.size(); ++k) {
    if (depth == types[k].depth1 && eta*types[k].z > 0) {
      id1 = k; break;
    }
  }
  return id1;
}

std::vector<std::pair<std::string,std::string> > SimHitsValidationHE::getHistogramTypes() {

  std::vector<std::pair<std::string,std::string> > divisions;
  std::pair<std::string,std::string> names;
  char                               name1[20], name2[20];
  SimHitsValidationHE::idType        type;

  for (int depth=0; depth<maxDepth_; ++depth) {
    sprintf (name1, "HE%d+z", depth);
    sprintf (name2, "HE +z depth%d", depth+1);
    names = std::pair<std::string,std::string>(std::string(name1),std::string(name2));
    type  = SimHitsValidationHE::idType(1,depth+1,depth+1);
    divisions.push_back(names);
    types.push_back(type);
    sprintf (name1, "HE%d-z", depth);
    sprintf (name2, "HE -z depth%d", depth+1);
    names = std::pair<std::string,std::string>(std::string(name1),std::string(name2));
    type  = SimHitsValidationHE::idType(-1,depth+1,depth+1);
    divisions.push_back(names);
    types.push_back(type);
  }

  return divisions;
    
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(SimHitsValidationHE);


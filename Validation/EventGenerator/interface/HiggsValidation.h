#ifndef HiggsValidation_H
#define HiggsValidation_H

/*class HiggsValidation
 *  
 *  Class to fill Event Generator dqm monitor elements; works on HepMCProduct
 *
 *  $Date: 2012/08/12 16:13:28 $
 *  $Revision: 1.1 $
 *
 */
#include <iostream>

// framework & common header files
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

//DQM services
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
//#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include "TLorentzVector.h"

#include "Validation/EventGenerator/interface/WeightManager.h"

class HiggsValidation : public edm::EDAnalyzer {
 public:
  explicit HiggsValidation(const edm::ParameterSet&);
  virtual ~HiggsValidation();
  virtual void beginJob();
  virtual void endJob();  
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void beginRun(const edm::Run&, const edm::EventSetup&);
  virtual void endRun(const edm::Run&, const edm::EventSetup&);
  
 private:


  class MonitoredDecays {
  public:
    
    MonitoredDecays(const edm::ParameterSet& iConfig){
      fillMap();
      std::vector<std::string> input = iConfig.getParameter<std::vector<std::string> >("monitorDecays");
      for(std::vector<std::string>::const_iterator i = input.begin(); i!= input.end(); ++i){
	fill(*i);
      }
    }

    ~MonitoredDecays(){};
        
    size_t position(int pid1,int pid2){
      if(abs(pid1) == 14 || abs(pid1) == 16) pid1 = 12;
      if(abs(pid2) == 14 || abs(pid2) == 16) pid2 = 12;
      for(size_t i = 0; i < channels.size(); ++i){
	if((channels[i].first == abs(pid1) && channels[i].second == abs(pid2)) || 
	   (channels[i].first == abs(pid2) && channels[i].second == abs(pid1))) return i+1;
      }
      return undetermined();//channels.size()+1;
    }

    size_t size(){return channels.size() + 2;}
    size_t undetermined(){return 0;}
    size_t stable(){return size();}
    std::string channel(size_t i){
      if(i == 0) return "?";
      if(i == channels.size()+1) return "Stable";
      return convert(channels[i-1].first)+convert(channels[i-1].second);
    }

    int convert(std::string s){
      if( namePidMap.count(s)){
        return namePidMap[s];
      }
      return 0;
    }

    std::string convert(int pid){
      pid = abs(pid);
      if(pid == 14 || pid == 16) pid = 12;
      for(std::map<std::string,int>::const_iterator i = namePidMap.begin(); i!= namePidMap.end(); ++i) {
        if(i->second == pid) return i->first;
      }
      return "not found";
    }

    unsigned int NDecayParticles(){return nparticles_;}

    int isDecayParticle(int pid){
      int idx=0;
      for(std::map<std::string,int>::const_iterator i = namePidMap.begin(); i!= namePidMap.end(); ++i) {
        if(i->second == pid) return idx;
	idx++;
      }
      return -1;
    }

    std::string ConvertIndex(int index){
      int idx=0;
      for(std::map<std::string,int>::const_iterator i = namePidMap.begin(); i!= namePidMap.end(); ++i) {
        if(idx==index) return i->first;
        idx++;
      }
      return "unknown";
    }

  private:
    void fill(std::string s){
      size_t pos = s.find("+");
      std::string particle1 = s.substr(0,pos);
      std::string particle2 = s.substr(pos+1,s.length()-pos);
      std::pair<int,int> decay;
      decay.first  = convert(particle1);
      decay.second = convert(particle2);
      channels.push_back(decay);
    }
    
    void fillMap(){
      namePidMap["d"]     = 1;
      namePidMap["u"]     = 2;
      namePidMap["s"]     = 3;
      namePidMap["c"]     = 4;
      namePidMap["b"]     = 5;
      namePidMap["t"]     = 6;
      namePidMap["e"]     = 11;
      namePidMap["nu"]    = 12;
      namePidMap["mu"]    = 13;
      namePidMap["tau"]   = 15;
      namePidMap["gamma"] = 22;
      namePidMap["Z"]     = 23;
      namePidMap["W"]     = 24;
      nparticles_=0;
      for(std::map<std::string,int>::const_iterator i = namePidMap.begin(); i!= namePidMap.end(); ++i){
	nparticles_++;
      }
    }

    std::map<std::string,int> namePidMap;
    
    std::vector<std::pair<int,int> > channels;
    unsigned int nparticles_;

  };
  
  int findHiggsDecayChannel(const HepMC::GenParticle*,std::vector<HepMC::GenParticle*> &decayprod);
  std::string convert(int);
  
  WeightManager _wmanager;
  
  edm::InputTag hepmcCollection_;

  int particle_id;
  std::string particle_name;
  
  MonitoredDecays* monitoredDecays;
  
  /// PDT table
  edm::ESHandle<HepPDT::ParticleDataTable> fPDGTable ;
  
  ///ME's "container"
  DQMStore *dbe;

  MonitorElement *nEvt;
  MonitorElement *HiggsDecayChannels;
  
  MonitorElement *Higgs_pt;
  MonitorElement *Higgs_eta;
  MonitorElement *Higgs_mass;
  
  std::vector<MonitorElement*> HiggsDecayProd_pt;
  std::vector<MonitorElement*> HiggsDecayProd_eta;

};

#endif

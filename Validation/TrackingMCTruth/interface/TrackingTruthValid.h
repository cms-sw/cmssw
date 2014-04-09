#ifndef TrackingTruthValid_h
#define TrackingTruthValid_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <string>

class DQMStore;
class MonitorElement;
class TrackingParticle;

class TrackingTruthValid  : public  DQMEDAnalyzer {
 public:
  typedef std::vector<TrackingParticle> TrackingParticleCollection;
  //Constructor
  explicit TrackingTruthValid(const edm::ParameterSet& conf) ;
  //Destructor
  ~TrackingTruthValid(){} ;
  
  virtual void analyze(const edm::Event&, const edm::EventSetup& );

  void bookHistograms(DQMStore::IBooker & ibooker,const edm::Run& run, const edm::EventSetup& es);
  void beginJob(const edm::ParameterSet& conf);
  void endJob();
  
 private:
  bool runStandalone;
  std::string outputFile;
  
  DQMStore* dbe_;
  MonitorElement* meTPMass;
  MonitorElement* meTPCharge; 
  MonitorElement* meTPId;
  MonitorElement* meTPProc;
  MonitorElement* meTPAllHits;
  MonitorElement* meTPMatchedHits;
  MonitorElement* meTPPt;
  MonitorElement* meTPEta;
  MonitorElement* meTPPhi;
  MonitorElement* meTPVtxX;
  MonitorElement* meTPVtxY;
  MonitorElement* meTPVtxZ; 
  MonitorElement* meTPtip;
  MonitorElement* meTPlip;

  edm::EDGetTokenT<TrackingParticleCollection> vec_TrackingParticle_Token_;
};

#endif

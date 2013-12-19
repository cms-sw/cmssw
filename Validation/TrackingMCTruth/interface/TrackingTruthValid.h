#ifndef TrackingTruthValid_h
#define TrackingTruthValid_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include <string>

class DQMStore;
class MonitorElement;
class TrackingParticle;

class TrackingTruthValid  : public edm::EDAnalyzer {
 public:
  typedef std::vector<TrackingParticle> TrackingParticleCollection;
  //Constructor
  explicit TrackingTruthValid(const edm::ParameterSet& conf) ;
  //Destructor
  ~TrackingTruthValid(){} ;
  
  virtual void analyze(const edm::Event&, const edm::EventSetup& );

  void beginJob(const edm::ParameterSet& conf);
  void beginRun( const edm::Run&, const edm::EventSetup& ); 
  void endJob();
  
 private:
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

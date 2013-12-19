#ifndef TrackingTruthValid_h
#define TrackingTruthValid_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include <string>

class DQMStore;
class MonitorElement;

class TrackingTruthValid  : public edm::EDAnalyzer {
 public:
  //Constructor
  explicit TrackingTruthValid(const edm::ParameterSet& conf) ;
  //Destructor
  ~TrackingTruthValid(){} ;
  
  virtual void analyze(const edm::Event&, const edm::EventSetup& );

  void beginJob(const edm::ParameterSet& conf);
  void beginRun( const edm::Run&, const edm::EventSetup& ); 
  void endJob();
  
 private:
  DQMStore* dbe_;
  edm::ParameterSet conf_;
  std::string outputFile;
  edm::InputTag src_;
  
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
  
};

#endif

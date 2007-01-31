#ifndef CSCRecHitValidation_h
#define CSCRecHitValidation_h

// user include files

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "Validation/MuonCSCDigis/src/PSimHitMap.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "DataFormats/CSCRecHit/interface/CSCRecHit2D.h"


class CSCRecHitValidation : public edm::EDAnalyzer {
public:
  explicit CSCRecHitValidation(const edm::ParameterSet&);
  ~CSCRecHitValidation();
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void beginJob(const edm::EventSetup&){} 
  virtual void endJob() ;
 

 private:
  void plotResolution(const PSimHit & simHit, const CSCRecHit2D & recHit,
                      const CSCLayer * layer, int chamberType);

  const CSCLayer * findLayer(int detId) const;

  DaqMonitorBEInterface* dbe_;
  edm::InputTag theInputTag;
  std::string theOutputFile;
  PSimHitMap theSimHitMap;
  const CSCGeometry * theCSCGeometry;

  MonitorElement* theNPerEventPlot;
  MonitorElement* theResolutionPlots[10];
  MonitorElement* thePullPlots[10];
};

#endif


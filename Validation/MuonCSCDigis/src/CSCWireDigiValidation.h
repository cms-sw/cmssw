#ifndef CSCWireDigiValidation_H
#define CSCWireDigiValidation_H

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DataFormats/CSCDigi/interface/CSCWireDigi.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "Validation/MuonCSCDigis/src/PSimHitMap.h"

#include "Geometry/CSCGeometry/interface/CSCGeometry.h"

class CSCWireDigiValidation
{
public:
  CSCWireDigiValidation(const edm::ParameterSet&, DaqMonitorBEInterface* dbe,
                        const PSimHitMap & hitMap);
  ~CSCWireDigiValidation();
  void analyze(const edm::Event&, const edm::EventSetup&);
  void beginJob(const edm::EventSetup&) {}
  void endJob() {}

  void setGeometry(const CSCGeometry * geom) {theCSCGeometry = geom;}
 
  void plotResolution(const PSimHit & hit, const CSCWireDigi & digi,
                      const CSCLayer * layer, int chamberType);

  const CSCLayer * findLayer(int detId) const;

 private:
  DaqMonitorBEInterface* dbe_;
  edm::InputTag theInputTag;
  const PSimHitMap & theSimHitMap;
  const CSCGeometry * theCSCGeometry;


  MonitorElement* theTimeBinPlots[10];
  MonitorElement* theNDigisPerLayerPlots[10];
  MonitorElement* theResolutionPlots[10];
  MonitorElement* theNDigisPerEventPlot;
   
};

#endif


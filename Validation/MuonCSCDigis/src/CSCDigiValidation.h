#ifndef HCALDIGITESTER_H
#define HCALDIGITESTER_H

// user include files

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "Validation/MuonCSCDigis/src/CSCStripDigiValidation.h"
#include "Validation/MuonCSCDigis/src/CSCWireDigiValidation.h"
#include "Validation/MuonCSCDigis/src/CSCComparatorDigiValidation.h"
#include "Validation/MuonCSCDigis/src/PSimHitMap.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"


class CSCDigiValidation : public edm::EDAnalyzer {
public:
  explicit CSCDigiValidation(const edm::ParameterSet&);
  ~CSCDigiValidation();
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void beginJob(const edm::EventSetup&){} 
  virtual void endJob() ;
 

 private:
  DaqMonitorBEInterface* dbe_;
  std::string outputFile_;
  PSimHitMap theSimHitMap;
  CSCGeometry * theCSCGeometry;

  CSCStripDigiValidation      theStripDigiValidation;
  CSCWireDigiValidation       theWireDigiValidation;
  CSCComparatorDigiValidation theComparatorDigiValidation;

};

#endif


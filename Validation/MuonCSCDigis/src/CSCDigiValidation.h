#ifndef CSCDigiValidation_H
#define CSCDigiValidation_H

// user include files

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "SimMuon/MCTruth/interface/PSimHitMap.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"

class CSCStripDigiValidation;
class CSCWireDigiValidation;
class CSCComparatorDigiValidation;
class CSCALCTDigiValidation;
class CSCCLCTDigiValidation;

class CSCDigiValidation : public edm::EDAnalyzer {
public:
  explicit CSCDigiValidation(const edm::ParameterSet&);
  ~CSCDigiValidation();
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
 

 private:
  DQMStore* dbe_;
  std::string outputFile_;
  PSimHitMap theSimHitMap;
  CSCGeometry * theCSCGeometry;

  CSCStripDigiValidation      * theStripDigiValidation;
  CSCWireDigiValidation       * theWireDigiValidation;
  CSCComparatorDigiValidation * theComparatorDigiValidation;
  CSCALCTDigiValidation * theALCTDigiValidation;
  CSCCLCTDigiValidation * theCLCTDigiValidation;

};

#endif


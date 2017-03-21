#ifndef CSCDigiValidation_H
#define CSCDigiValidation_H

// user include files

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include <DQMServices/Core/interface/DQMStore.h>
#include <DQMServices/Core/interface/DQMEDAnalyzer.h>

#include "SimMuon/MCTruth/interface/PSimHitMap.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"

class CSCStripDigiValidation;
class CSCWireDigiValidation;
class CSCComparatorDigiValidation;
class CSCALCTDigiValidation;
class CSCCLCTDigiValidation;

class CSCDigiValidation : public DQMEDAnalyzer {
public:
  explicit CSCDigiValidation(const edm::ParameterSet&);
  ~CSCDigiValidation();
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  bool doSim_;
  PSimHitMap theSimHitMap;
  CSCGeometry* theCSCGeometry;

  CSCStripDigiValidation*      theStripDigiValidation;
  CSCWireDigiValidation*       theWireDigiValidation;
  CSCComparatorDigiValidation* theComparatorDigiValidation;
  CSCALCTDigiValidation*       theALCTDigiValidation;
  CSCCLCTDigiValidation*       theCLCTDigiValidation;
};

#endif


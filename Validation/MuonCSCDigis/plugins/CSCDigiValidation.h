#ifndef Validation_MuonCSCDigis_CSCDigiValidation_H
#define Validation_MuonCSCDigis_CSCDigiValidation_H

// user include files

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "SimMuon/MCTruth/interface/PSimHitMap.h"

class CSCStripDigiValidation;
class CSCWireDigiValidation;
class CSCComparatorDigiValidation;
class CSCALCTDigiValidation;
class CSCCLCTDigiValidation;

class CSCDigiValidation : public DQMEDAnalyzer {
public:
  explicit CSCDigiValidation(const edm::ParameterSet &);
  ~CSCDigiValidation() override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void analyze(const edm::Event &, const edm::EventSetup &) override;

private:
  bool doSim_;
  PSimHitMap theSimHitMap;
  CSCGeometry *theCSCGeometry;

  std::unique_ptr<CSCStripDigiValidation> theStripDigiValidation;
  std::unique_ptr<CSCWireDigiValidation> theWireDigiValidation;
  std::unique_ptr<CSCComparatorDigiValidation> theComparatorDigiValidation;
  std::unique_ptr<CSCALCTDigiValidation> theALCTDigiValidation;
  std::unique_ptr<CSCCLCTDigiValidation> theCLCTDigiValidation;
};

#endif

#ifndef CSCRecHitValidation_h
#define CSCRecHitValidation_h

// user include files

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include <DQMServices/Core/interface/DQMStore.h>
#include <DQMServices/Core/interface/MonitorElement.h>
#include <DQMServices/Core/interface/DQMEDAnalyzer.h>

#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "SimMuon/MCTruth/interface/PSimHitMap.h"
#include "Validation/CSCRecHits/src/CSCRecHit2DValidation.h"
#include "Validation/CSCRecHits/src/CSCSegmentValidation.h"



class CSCRecHitValidation : public DQMEDAnalyzer {
public:
  explicit CSCRecHitValidation(const edm::ParameterSet&);
  ~CSCRecHitValidation();
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;

 private:
  PSimHitMap theSimHitMap;
  const CSCGeometry * theCSCGeometry;

  CSCRecHit2DValidation * the2DValidation;
  CSCSegmentValidation * theSegmentValidation;
};

#endif


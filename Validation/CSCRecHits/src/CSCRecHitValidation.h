#ifndef CSCRecHitValidation_h
#define CSCRecHitValidation_h

// user include files

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "SimMuon/MCTruth/interface/PSimHitMap.h"
#include "Validation/CSCRecHits/src/CSCRecHit2DValidation.h"
#include "Validation/CSCRecHits/src/CSCSegmentValidation.h"

class CSCRecHitValidation : public DQMEDAnalyzer {
public:
  explicit CSCRecHitValidation(const edm::ParameterSet &);
  ~CSCRecHitValidation() override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void analyze(const edm::Event &, const edm::EventSetup &) override;

private:
  PSimHitMap theSimHitMap;
  edm::ESGetToken<CSCGeometry, MuonGeometryRecord> geomToken_;

  CSCRecHit2DValidation *the2DValidation;
  CSCSegmentValidation *theSegmentValidation;
};

#endif

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "SimMuon/MCTruth/interface/PSimHitMap.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Validation/CSCRecHits/interface/CSCRecHit2DValidation.h"
#include "Validation/CSCRecHits/interface/CSCSegmentValidation.h"

class CSCRecHitValidation : public DQMEDAnalyzer {
public:
  explicit CSCRecHitValidation(const edm::ParameterSet &);
  ~CSCRecHitValidation() override{};
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void analyze(const edm::Event &, const edm::EventSetup &) override;

private:
  PSimHitMap theSimHitMap;
  edm::ESGetToken<CSCGeometry, MuonGeometryRecord> geomToken_;

  std::unique_ptr<CSCRecHit2DValidation> the2DValidation;
  std::unique_ptr<CSCSegmentValidation> theSegmentValidation;
};

DEFINE_FWK_MODULE(CSCRecHitValidation);

CSCRecHitValidation::CSCRecHitValidation(const edm::ParameterSet &ps)
    : theSimHitMap(ps.getParameter<edm::InputTag>("simHitsTag"), consumesCollector()),
      the2DValidation(nullptr),
      theSegmentValidation(nullptr) {
  the2DValidation = std::make_unique<CSCRecHit2DValidation>(ps, consumesCollector());
  theSegmentValidation = std::make_unique<CSCSegmentValidation>(ps, consumesCollector());
  geomToken_ = esConsumes<CSCGeometry, MuonGeometryRecord>();
}

void CSCRecHitValidation::bookHistograms(DQMStore::IBooker &iBooker, edm::Run const &iRun, edm::EventSetup const &) {
  iBooker.setCurrentFolder("CSCRecHitsV/CSCRecHitTask");

  the2DValidation->bookHistograms(iBooker);
  theSegmentValidation->bookHistograms(iBooker);
}

void CSCRecHitValidation::analyze(const edm::Event &e, const edm::EventSetup &eventSetup) {
  theSimHitMap.fill(e);

  // find the geometry & conditions for this event
  const CSCGeometry *theCSCGeometry = &eventSetup.getData(geomToken_);

  the2DValidation->setGeometry(theCSCGeometry);
  the2DValidation->setSimHitMap(&theSimHitMap);

  theSegmentValidation->setGeometry(theCSCGeometry);
  theSegmentValidation->setSimHitMap(&theSimHitMap);

  the2DValidation->analyze(e, eventSetup);
  theSegmentValidation->analyze(e, eventSetup);
}

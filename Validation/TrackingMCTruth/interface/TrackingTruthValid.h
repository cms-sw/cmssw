#ifndef TrackingTruthValid_h
#define TrackingTruthValid_h

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include <string>

class TrackingParticle;

class TrackingTruthValid : public DQMEDAnalyzer {
public:
  typedef std::vector<TrackingParticle> TrackingParticleCollection;
  // Constructor
  explicit TrackingTruthValid(const edm::ParameterSet &conf);
  // Destructor
  ~TrackingTruthValid() override{};

  void analyze(const edm::Event &, const edm::EventSetup &) override;

  void bookHistograms(DQMStore::IBooker &ibooker, const edm::Run &run, const edm::EventSetup &es) override;
  void endJob() override;

private:
  bool runStandalone;
  std::string outputFile;

  DQMStore *dbe_;
  MonitorElement *meTPMass;
  MonitorElement *meTPCharge;
  MonitorElement *meTPId;
  MonitorElement *meTPProc;
  MonitorElement *meTPAllHits;
  MonitorElement *meTPMatchedHits;
  MonitorElement *meTPPt;
  MonitorElement *meTPEta;
  MonitorElement *meTPPhi;
  MonitorElement *meTPVtxX;
  MonitorElement *meTPVtxY;
  MonitorElement *meTPVtxZ;
  MonitorElement *meTPtip;
  MonitorElement *meTPlip;

  edm::EDGetTokenT<TrackingParticleCollection> vec_TrackingParticle_Token_;
};

#endif

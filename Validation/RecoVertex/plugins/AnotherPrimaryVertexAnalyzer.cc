// -*- C++ -*-
//
// Package:    Validation/RecoVertex
// Class:      AnotherPrimaryVertexAnalyzer
//
/**\class AnotherPrimaryVertexAnalyzer AnotherPrimaryVertexAnalyzer.cc Validation/RecoVertex/plugins/AnotherPrimaryVertexAnalyzer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Andrea Venturi
//         Created:  Mon Oct 27 17:37:53 CET 2008
//
//

// system include files
#include <memory>

// user include files

#include <vector>
#include <map>
#include <limits>
#include <string>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Utilities/interface/InputTag.h"

#include "Validation/RecoVertex/interface/VertexHistogramMaker.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "CommonTools/TriggerUtils/interface/PrescaleWeightProvider.h"

//
// class decleration
//

class AnotherPrimaryVertexAnalyzer : public edm::one::EDAnalyzer<edm::one::WatchRuns, edm::one::SharedResources> {
public:
  explicit AnotherPrimaryVertexAnalyzer(const edm::ParameterSet&);
  ~AnotherPrimaryVertexAnalyzer() override;

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void beginRun(const edm::Run&, const edm::EventSetup&) override;
  void endRun(const edm::Run&, const edm::EventSetup&) override;
  void endJob() override;

  // ----------member data ---------------------------

  VertexHistogramMaker _vhm;
  edm::EDGetTokenT<reco::VertexCollection> _recoVertexCollectionToken;
  bool _firstOnly;

  std::unique_ptr<PrescaleWeightProvider> _weightprov;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
AnotherPrimaryVertexAnalyzer::AnotherPrimaryVertexAnalyzer(const edm::ParameterSet& iConfig)
    : _vhm(iConfig.getParameter<edm::ParameterSet>("vHistogramMakerPSet"), consumesCollector()),
      _recoVertexCollectionToken(consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("pvCollection"))),
      _firstOnly(iConfig.getUntrackedParameter<bool>("firstOnly", false)),
      _weightprov(
          iConfig.getParameter<bool>("usePrescaleWeight")
              ? new PrescaleWeightProvider(
                    iConfig.getParameter<edm::ParameterSet>("prescaleWeightProviderPSet"), consumesCollector(), *this)
              : nullptr) {
  //now do what ever initialization is needed
  usesResource(TFileService::kSharedResource);
  //
  _vhm.book();
}

AnotherPrimaryVertexAnalyzer::~AnotherPrimaryVertexAnalyzer() {}

//
// member functions
//

// ------------ method called to for each event  ------------
void AnotherPrimaryVertexAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // compute event weight
  auto const weight = _weightprov ? _weightprov->prescaleWeight<double>(iEvent, iSetup) : 1.;

  // get PV
  edm::Handle<reco::VertexCollection> pvcoll;
  iEvent.getByToken(_recoVertexCollectionToken, pvcoll);

  if (_firstOnly) {
    reco::VertexCollection firstpv;
    if (!pvcoll->empty())
      firstpv.push_back((*pvcoll)[0]);
    _vhm.fill(iEvent, firstpv, weight);
  } else {
    _vhm.fill(iEvent, *pvcoll, weight);
  }
}

// ------------ method called once each job just before starting event loop  ------------
void AnotherPrimaryVertexAnalyzer::beginJob() {}

void AnotherPrimaryVertexAnalyzer::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {
  _vhm.beginRun(iRun);

  if (_weightprov)
    _weightprov->initRun(iRun, iSetup);
}

void AnotherPrimaryVertexAnalyzer::endRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {}
// ------------ method called once each job just after ending the event loop  ------------
void AnotherPrimaryVertexAnalyzer::endJob() {}

//define this as a plug-in
DEFINE_FWK_MODULE(AnotherPrimaryVertexAnalyzer);

// -*- C++ -*-
//
/**\class SiOuterTracker Phase2OTValidateTTStub.cc
 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:
//

// system include files
#include <memory>
#include <numeric>
#include <vector>

// user include files
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/L1TrackTrigger/interface/TTStub.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

class Phase2OTValidateTTStub : public DQMEDAnalyzer {
public:
  explicit Phase2OTValidateTTStub(const edm::ParameterSet &);
  ~Phase2OTValidateTTStub() override;
  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void dqmBeginRun(const edm::Run &iRun, const edm::EventSetup &iSetup) override;
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);
  // TTStub stacks
  // Global position of the stubs
  MonitorElement *Stub_RZ = nullptr;  // TTStub #rho vs. z

private:
  edm::ParameterSet conf_;
  edm::EDGetTokenT<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>> tagTTStubsToken_;
  std::string topFolderName_;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topoToken_;
  const TrackerGeometry *tkGeom_ = nullptr;
  const TrackerTopology *tTopo_ = nullptr;
};

// constructors and destructor
Phase2OTValidateTTStub::Phase2OTValidateTTStub(const edm::ParameterSet &iConfig)
    : conf_(iConfig),
      geomToken_(esConsumes<TrackerGeometry, TrackerDigiGeometryRecord, edm::Transition::BeginRun>()),
      topoToken_(esConsumes<TrackerTopology, TrackerTopologyRcd, edm::Transition::BeginRun>()) {
  // now do what ever initialization is needed
  topFolderName_ = conf_.getParameter<std::string>("TopFolderName");
  tagTTStubsToken_ =
      consumes<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>>(conf_.getParameter<edm::InputTag>("TTStubs"));
}

Phase2OTValidateTTStub::~Phase2OTValidateTTStub() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

void Phase2OTValidateTTStub::dqmBeginRun(const edm::Run &iRun, const edm::EventSetup &iSetup) {
  tkGeom_ = &(iSetup.getData(geomToken_));
  tTopo_ = &(iSetup.getData(topoToken_));
}
// member functions

// ------------ method called for each event  ------------
void Phase2OTValidateTTStub::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  /// Track Trigger Stubs
  edm::Handle<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>> Phase2TrackerDigiTTStubHandle;
  iEvent.getByToken(tagTTStubsToken_, Phase2TrackerDigiTTStubHandle);

  /// Loop over input Stubs
  typename edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>::const_iterator inputIter;
  typename edmNew::DetSet<TTStub<Ref_Phase2TrackerDigi_>>::const_iterator contentIter;
  // Adding protection
  if (!Phase2TrackerDigiTTStubHandle.isValid())
    return;

  for (inputIter = Phase2TrackerDigiTTStubHandle->begin(); inputIter != Phase2TrackerDigiTTStubHandle->end();
       ++inputIter) {
    for (contentIter = inputIter->begin(); contentIter != inputIter->end(); ++contentIter) {
      /// Make reference stub
      edm::Ref<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>, TTStub<Ref_Phase2TrackerDigi_>> tempStubRef =
          edmNew::makeRefTo(Phase2TrackerDigiTTStubHandle, contentIter);

      /// Get det ID (place of the stub)
      //  tempStubRef->getDetId() gives the stackDetId, not rawId
      DetId detIdStub = tkGeom_->idToDet((tempStubRef->clusterRef(0))->getDetId())->geographicalId();

      /// Get trigger displacement/offset
      //double rawBend = tempStubRef->rawBend();
      //double bendOffset = tempStubRef->bendOffset();

      /// Define position stub by position inner cluster
      MeasurementPoint mp = (tempStubRef->clusterRef(0))->findAverageLocalCoordinates();
      const GeomDet *theGeomDet = tkGeom_->idToDet(detIdStub);
      Global3DPoint posStub = theGeomDet->surface().toGlobal(theGeomDet->topology().localPosition(mp));

      Stub_RZ->Fill(posStub.z(), posStub.perp());
    }
  }
}  // end of method

// ------------ method called when starting to processes a run  ------------
void Phase2OTValidateTTStub::bookHistograms(DQMStore::IBooker &iBooker,
                                            edm::Run const &run,
                                            edm::EventSetup const &es) {
  std::string HistoName;
  iBooker.setCurrentFolder(topFolderName_);
  edm::ParameterSet psTTStub_RZ = conf_.getParameter<edm::ParameterSet>("TH2TTStub_RZ");
  HistoName = "Stub_RZ";
  Stub_RZ = iBooker.book2D(HistoName,
                           HistoName,
                           psTTStub_RZ.getParameter<int32_t>("Nbinsx"),
                           psTTStub_RZ.getParameter<double>("xmin"),
                           psTTStub_RZ.getParameter<double>("xmax"),
                           psTTStub_RZ.getParameter<int32_t>("Nbinsy"),
                           psTTStub_RZ.getParameter<double>("ymin"),
                           psTTStub_RZ.getParameter<double>("ymax"));
}

void Phase2OTValidateTTStub::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  // Phase2OTValidateTTStub
  edm::ParameterSetDescription desc;
  {
    edm::ParameterSetDescription psd0;
    psd0.add<int>("Nbinsx", 900);
    psd0.add<double>("xmax", 300);
    psd0.add<double>("xmin", -300);
    psd0.add<int>("Nbinsy", 900);
    psd0.add<double>("ymax", 120);
    psd0.add<double>("ymin", 0);
    desc.add<edm::ParameterSetDescription>("TH2TTStub_RZ", psd0);
  }
  desc.add<std::string>("TopFolderName", "TrackerPhase2OTStubV");
  desc.add<edm::InputTag>("TTStubs", edm::InputTag("TTStubsFromPhase2TrackerDigis", "StubAccepted"));
  descriptions.add("Phase2OTValidateTTStub", desc);
  // or use the following to generate the label from the module's C++ type
  //descriptions.addWithDefaultLabel(desc);
}
DEFINE_FWK_MODULE(Phase2OTValidateTTStub);

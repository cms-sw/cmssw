// -*- C++ -*-
//
// Package:    MuonSegmentProducer
// Class:      MuonSegmentProducer
// 
/**\class MuonSegmentProducer MuonSegmentProducer.cc SUSYBSMAnalysis/MuonSegmentProducer/src/MuonSegmentProducer.cc

 Description: Producer muon segments with global position info to be used in FWLite

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Rizzi Andrea
// Reworked and Ported to CMSSW_3_0_0 by Christophe Delaere
//         Created:  Wed Oct 10 12:01:28 CEST 2007
// $Id: MuonSegmentProducer.cc,v 1.1 2012/04/27 20:49:41 farrell3 Exp $
//
//

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "AnalysisDataFormats/SUSYBSMObjects/interface/MuonSegment.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"
#include "Geometry/DTGeometry/interface/DTChamber.h"
#include "Geometry/DTGeometry/interface/DTSuperLayer.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

#include <vector>
#include <iostream>

//                                                                                                                                                                                 
// class decleration                                                                                                                                                               
//                                                                                                                                                                                 
class MuonSegmentProducer : public edm::EDProducer {
public:
  explicit MuonSegmentProducer(const edm::ParameterSet&);
  ~MuonSegmentProducer();

private:
  virtual void beginJob() ;
  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

  edm::EDGetTokenT< CSCSegmentCollection > m_cscSegmentToken;
  edm::EDGetTokenT< DTRecSegment4DCollection > m_dtSegmentToken;
};

using namespace susybsm;

MuonSegmentProducer::MuonSegmentProducer(const edm::ParameterSet& iConfig) {
  using namespace edm;
  using namespace std;


  m_cscSegmentToken = consumes< CSCSegmentCollection >( iConfig.getParameter<edm::InputTag>("CSCSegments" ) );
  m_dtSegmentToken  = consumes< DTRecSegment4DCollection >(iConfig.getParameter<edm::InputTag>("DTSegments" ) );

  produces<susybsm::MuonSegmentCollection >();
}

MuonSegmentProducer::~MuonSegmentProducer() {
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void
MuonSegmentProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {

  using namespace edm;
  using namespace std;
  using namespace susybsm;

  susybsm::MuonSegmentCollection* segments = new susybsm::MuonSegmentCollection;
  std::unique_ptr<susybsm::MuonSegmentCollection> resultSeg(segments);

  edm::ESHandle<DTGeometry> dtGeom;
  iSetup.get<MuonGeometryRecord>().get(dtGeom);

  edm::ESHandle<CSCGeometry> cscGeom;
  iSetup.get<MuonGeometryRecord>().get(cscGeom);

  edm::Handle<DTRecSegment4DCollection> dtSegments;
  iEvent.getByToken(m_dtSegmentToken, dtSegments);

  for (unsigned int d=0; d<dtSegments->size(); d++) {
    DTRecSegment4DRef SegRef  = DTRecSegment4DRef( dtSegments, d );
    MuonSegment muonSegment;
    muonSegment.setDTSegmentRef(SegRef);

    const GeomDet* dtDet = dtGeom->idToDet(SegRef->geographicalId());
    GlobalPoint point = dtDet->toGlobal(SegRef->localPosition());
    muonSegment.setGP(point);
    segments->push_back(muonSegment);
  }

  edm::Handle<CSCSegmentCollection> cscSegments;
  iEvent.getByToken(m_cscSegmentToken, cscSegments);

  for (unsigned int c=0; c<cscSegments->size(); c++) {
    CSCSegmentRef SegRef  = CSCSegmentRef( cscSegments, c );
    MuonSegment muonSegment;
    muonSegment.setCSCSegmentRef(SegRef);

    const GeomDet* cscDet = cscGeom->idToDet(SegRef->geographicalId());
    GlobalPoint point = cscDet->toGlobal(SegRef->localPosition());
    muonSegment.setGP(point);
    segments->push_back(muonSegment);
  }

  edm::OrphanHandle<susybsm::MuonSegmentCollection> putHandleSeg = iEvent.put(std::move(resultSeg));
}

// ------------ method called once each job just before starting event loop  ------------
void 
MuonSegmentProducer::beginJob() {
}

// ------------ method called once each job just after ending the event loop  ------------
void 
MuonSegmentProducer::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(MuonSegmentProducer);




















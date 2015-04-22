#include "SimMuon/GEMDigitizer/interface/GEMPadDigiProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/Common/interface/Handle.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <string>
#include <map>
#include <vector>


GEMPadDigiProducer::GEMPadDigiProducer(const edm::ParameterSet& ps)
: geometry_(nullptr)
{
  digis_ = ps.getParameter<edm::InputTag>("InputCollection");

  digi_token_ = consumes<GEMDigiCollection>(digis_);

  produces<GEMPadDigiCollection>();
  consumes<GEMDigiCollection>(digis_);
}


GEMPadDigiProducer::~GEMPadDigiProducer()
{}


void GEMPadDigiProducer::beginRun(const edm::Run& run, const edm::EventSetup& eventSetup)
{
  edm::ESHandle<GEMGeometry> hGeom;
  eventSetup.get<MuonGeometryRecord>().get(hGeom);
  geometry_ = &*hGeom;
}


void GEMPadDigiProducer::produce(edm::Event& e, const edm::EventSetup& eventSetup)
{
  edm::Handle<GEMDigiCollection> hdigis;
  e.getByToken(digi_token_, hdigis);

  // Create empty output
  std::auto_ptr<GEMPadDigiCollection> pPads(new GEMPadDigiCollection());

  // build the pads
  buildPads(*(hdigis.product()), *pPads);

  // store them in the event
  e.put(pPads);
}


void GEMPadDigiProducer::buildPads(const GEMDigiCollection &det_digis, GEMPadDigiCollection &out_pads)
{
  auto etaPartitions = geometry_->etaPartitions();
  for(auto p: etaPartitions)
  {
    // set of <pad, bx> pairs, sorted first by pad then by bx
    std::set<std::pair<int, int> > proto_pads;
  
    // walk over digis in this partition, 
    // and stuff them into a set of unique pads (equivalent of OR operation)
    auto digis = det_digis.get(p->id());
    for (auto d = digis.first; d != digis.second; ++d)
    {
      int pad_num = 1 + static_cast<int>( p->padOfStrip(d->strip()) );
      auto pad = std::make_pair(pad_num, d->bx());
      proto_pads.insert(pad);
    }
  
    // in the future, do some dead-time handling
    // emulateDeadTime(proto_pads)
  
    // fill the output collections
    for (auto & d: proto_pads)
    {
      GEMPadDigi pad_digi(d.first, d.second);
      out_pads.insertDigi(p->id(), pad_digi);
    }
  }
}

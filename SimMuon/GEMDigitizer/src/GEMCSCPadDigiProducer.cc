#include "SimMuon/GEMDigitizer/interface/GEMCSCPadDigiProducer.h"

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/Common/interface/Handle.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <string>
#include <map>
#include <vector>


GEMCSCPadDigiProducer::GEMCSCPadDigiProducer(const edm::ParameterSet& ps)
: geometry_(nullptr)
{
  produces<GEMCSCPadDigiCollection>();
  produces<GEMCSCPadDigiCollection>("Coincidence");

  input_ = ps.getParameter<edm::InputTag>("InputCollection");
  maxDeltaBX_ = ps.getParameter<int>("maxDeltaBX");
}


GEMCSCPadDigiProducer::~GEMCSCPadDigiProducer()
{}


void GEMCSCPadDigiProducer::produce(edm::Event& e, const edm::EventSetup& eventSetup)
{
  // set geometry
  edm::ESHandle<GEMGeometry> hGeom;
  eventSetup.get<MuonGeometryRecord>().get( hGeom );
  geometry_ = &*hGeom;

  edm::Handle<GEMDigiCollection> hdigis;
  e.getByLabel(input_, hdigis);

  // Create empty output
  std::auto_ptr<GEMCSCPadDigiCollection> pPads(new GEMCSCPadDigiCollection());
  std::auto_ptr<GEMCSCPadDigiCollection> pCoPads(new GEMCSCPadDigiCollection());

  // build the pads
  buildPads(*(hdigis.product()), *pPads, *pCoPads);

  // store them in the event
  e.put(pPads);
  e.put(pCoPads, "Coincidence");
}


void GEMCSCPadDigiProducer::buildPads(const GEMDigiCollection &det_digis,
    GEMCSCPadDigiCollection &out_pads, GEMCSCPadDigiCollection &out_co_pads)
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
      GEMCSCPadDigi pad_digi(d.first, d.second);
      out_pads.insertDigi(p->id(), pad_digi);
    }
  }

  // build coincidences
  for (auto det_range = out_pads.begin(); det_range != out_pads.end(); ++det_range)
  {
    const GEMDetId& id = (*det_range).first;

    // all coincidences detIDs will have layer=1
    if (id.layer() != 1) continue;

    // find the corresponding id with layer=2
    GEMDetId co_id(id.region(), id.ring(), id.station(), 2, id.chamber(), id.roll());

    auto co_pads_range = out_pads.get(co_id);
    // empty range = no possible coincidence pads
    if (co_pads_range.first == co_pads_range.second) continue;

    // now let's correlate the pads in two layers of this partition
    const auto& pads_range = (*det_range).second;
    for (auto p = pads_range.first; p != pads_range.second; ++p)
    {
      for (auto co_p = co_pads_range.first; co_p != co_pads_range.second; ++co_p)
      {
        // check the match!
        if (p->pad() != co_p->pad() || std::abs(p->bx() - co_p->bx()) > maxDeltaBX_ ) continue;

        // always use layer1 pad's BX as a copad's BX
        GEMCSCPadDigi co_pad_digi(p->pad(), p->bx());
        out_co_pads.insertDigi(id, co_pad_digi);
      }
    }
  }
}

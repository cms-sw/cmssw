#include "SimMuon/GEMDigitizer/interface/ME0PadDigiProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/Common/interface/Handle.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/GEMGeometry/interface/ME0Geometry.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <set>


ME0PadDigiProducer::ME0PadDigiProducer(const edm::ParameterSet& ps)
: geometry_(nullptr)
{
  digis_ = ps.getParameter<edm::InputTag>("InputCollection");

  digi_token_ = consumes<ME0DigiPreRecoCollection>(digis_);

  produces<ME0PadDigiCollection>();
  consumes<ME0DigiPreRecoCollection>(digis_);
}


ME0PadDigiProducer::~ME0PadDigiProducer()
{}


void ME0PadDigiProducer::beginRun(const edm::Run& run, const edm::EventSetup& eventSetup)
{
  edm::ESHandle<ME0Geometry> hGeom;
  eventSetup.get<MuonGeometryRecord>().get(hGeom);
  geometry_ = &*hGeom;
}


void ME0PadDigiProducer::produce(edm::Event& e, const edm::EventSetup& eventSetup)
{
  edm::Handle<ME0DigiPreRecoCollection> hdigis;
  e.getByToken(digi_token_, hdigis);

  // Create empty output
  std::unique_ptr<ME0PadDigiCollection> pPads(new ME0PadDigiCollection());

  // build the pads
  buildPads(*(hdigis.product()), *pPads);

  // store them in the event
  e.put(std::move(pPads));
}


void ME0PadDigiProducer::buildPads(const ME0DigiPreRecoCollection &det_digis, ME0PadDigiCollection &out_pads) const
{
  auto etaPartitions = geometry_->etaPartitions();
  for(const auto& p: etaPartitions)
  {
    // set of <pad, bx> pairs, sorted first by pad then by bx
    std::set<std::pair<int, int> > proto_pads;

    // walk over digis in this partition,
    // and stuff them into a set of unique pads (equivalent of OR operation)
    auto digis = det_digis.get(p->id());
    for (auto d = digis.first; d != digis.second; ++d)
    {
      int strip_num = p->strip(LocalPoint(d->x(),d->y(),0));
      int pad_num = 1 + static_cast<int>( p->padOfStrip(strip_num) );
      auto pad = std::make_pair(pad_num, d->tof());
      proto_pads.insert(pad);
    }

    // fill the output collections
    for (auto & d: proto_pads)
    {
      ME0PadDigi pad_digi(d.first, d.second);
      out_pads.insertDigi(p->id(), pad_digi);
    }
  }
}

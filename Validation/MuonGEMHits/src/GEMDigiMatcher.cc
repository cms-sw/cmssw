#include "Validation/MuonGEMHits/interface/GEMDigiMatcher.h"
#include "Validation/MuonGEMHits/interface//SimHitMatcher.h"

#include "DataFormats/MuonDetId/interface/GEMDetId.h"

using namespace std;
using namespace matching;


GEMDigiMatcher::GEMDigiMatcher(SimHitMatcher& sh)
: DigiMatcher(sh)
{
  gemDigiInput_ = conf().getUntrackedParameter<edm::InputTag>("gemDigiInput",
      edm::InputTag("simMuonGEMDigis"));
  gemPadDigiInput_ = conf().getUntrackedParameter<edm::InputTag>("gemPadDigiInput",
      edm::InputTag("simMuonGEMCSCPadDigis"));
  gemCoPadDigiInput_ = conf().getUntrackedParameter<edm::InputTag>("gemCoPadDigiInput",
      edm::InputTag("simMuonGEMCSCPadDigis", "Coincidence"));

  minBXGEM_ = conf().getUntrackedParameter<int>("minBXGEM", -1);
  maxBXGEM_ = conf().getUntrackedParameter<int>("maxBXGEM", 1);

  matchDeltaStrip_ = conf().getUntrackedParameter<int>("matchDeltaStripGEM", 1);

  setVerbose(conf().getUntrackedParameter<int>("verboseGEMDigi", 0));

  if (! (gemDigiInput_.label().empty() ||
         gemPadDigiInput_.label().empty() ||
         gemCoPadDigiInput_.label().empty())
     )
  {
    init();
  }
}

GEMDigiMatcher::~GEMDigiMatcher() {}


void
GEMDigiMatcher::init()
{
  edm::Handle<GEMDigiCollection> gem_digis;
  event().getByLabel(gemDigiInput_, gem_digis);
  matchDigisToSimTrack(*gem_digis.product());

  edm::Handle<GEMCSCPadDigiCollection> gem_pads;
  event().getByLabel(gemPadDigiInput_, gem_pads);
  matchPadsToSimTrack(*gem_pads.product());

  edm::Handle<GEMCSCPadDigiCollection> gem_co_pads;
  event().getByLabel(gemPadDigiInput_, gem_co_pads);
  matchCoPadsToSimTrack(*gem_co_pads.product());
}


void
GEMDigiMatcher::matchDigisToSimTrack(const GEMDigiCollection& digis)
{
  auto det_ids = simhit_matcher_->detIdsGEM();
  for (auto id: det_ids)
  {
    GEMDetId p_id(id);
    GEMDetId superch_id(p_id.region(), p_id.ring(), p_id.station(), 1, p_id.chamber(), 0);

    auto hit_strips = simhit_matcher_->hitStripsInDetId(id, matchDeltaStrip_);
    if (verbose())
    {
      cout<<"hit_strips_fat ";
      copy(hit_strips.begin(), hit_strips.end(), ostream_iterator<int>(cout, " "));
      cout<<endl;
    }

    auto digis_in_det = digis.get(GEMDetId(id));

    for (auto d = digis_in_det.first; d != digis_in_det.second; ++d)
    {
      if (verbose()) cout<<"gdigi "<<p_id<<" "<<*d<<endl;
      // check that the digi is within BX range
      if (d->bx() < minBXGEM_ || d->bx() > maxBXGEM_) continue;
      // check that it matches a strip that was hit by SimHits from our track
      if (hit_strips.find(d->strip()) == hit_strips.end()) continue;
      if (verbose()) cout<<"oki"<<endl;

      auto mydigi = make_digi(id, d->strip(), d->bx(), GEM_STRIP);
      detid_to_digis_[id].push_back(mydigi);
      chamber_to_digis_[ p_id.chamberId().rawId() ].push_back(mydigi);
      superchamber_to_digis_[ superch_id() ].push_back(mydigi);

      //int pad_num = 1 + static_cast<int>( roll->padOfStrip(d->strip()) ); // d->strip() is int
      //digi_map[ make_pair(pad_num, d->bx()) ].push_back( d->strip() );
    }
  }
}


void
GEMDigiMatcher::matchPadsToSimTrack(const GEMCSCPadDigiCollection& pads)
{
  auto det_ids = simhit_matcher_->detIdsGEM();
  for (auto id: det_ids)
  {
    GEMDetId p_id(id);
    GEMDetId superch_id(p_id.region(), p_id.ring(), p_id.station(), 1, p_id.chamber(), 0);

    auto hit_pads = simhit_matcher_->hitPadsInDetId(id);
    auto pads_in_det = pads.get(p_id);

    if (verbose())
    {
      cout<<"checkpads "<<hit_pads.size()<<" "<<std::distance(pads_in_det.first, pads_in_det.second)<<" hit_pads: ";
      copy(hit_pads.begin(), hit_pads.end(), ostream_iterator<int>(cout," "));
      cout<<endl;
    }

    for (auto pad = pads_in_det.first; pad != pads_in_det.second; ++pad)
    {
      if (verbose()) cout<<"chp "<<*pad<<endl;
      // check that the pad BX is within the range
      if (pad->bx() < minBXGEM_ || pad->bx() > maxBXGEM_) continue;
      if (verbose()) cout<<"chp1"<<endl;
      // check that it matches a pad that was hit by SimHits from our track
      if (hit_pads.find(pad->pad()) == hit_pads.end()) continue;
      if (verbose()) cout<<"chp2"<<endl;

      auto mydigi = make_digi(id, pad->pad(), pad->bx(), GEM_PAD);
      detid_to_pads_[id].push_back(mydigi);
      chamber_to_pads_[ p_id.chamberId().rawId() ].push_back(mydigi);
      superchamber_to_pads_[ superch_id() ].push_back(mydigi);
    }
  }
}


void
GEMDigiMatcher::matchCoPadsToSimTrack(const GEMCSCPadDigiCollection& co_pads)
{
  auto det_ids = simhit_matcher_->detIdsGEMCoincidences();
  for (auto id: det_ids)
  {
    GEMDetId p_id(id);
    GEMDetId superch_id(p_id.region(), p_id.ring(), p_id.station(), 1, p_id.chamber(), 0);

    auto hit_co_pads = simhit_matcher_->hitCoPadsInDetId(id);
    auto co_pads_in_det = co_pads.get(p_id);

    for (auto pad = co_pads_in_det.first; pad != co_pads_in_det.second; ++pad)
    {
      // check that the pad BX is within the range
      if (pad->bx() < minBXGEM_ || pad->bx() > maxBXGEM_) continue;
      // check that it matches a coincidence pad that was hit by SimHits from our track
      if (hit_co_pads.find(pad->pad()) == hit_co_pads.end()) continue;

      auto mydigi = make_digi(id, pad->pad(), pad->bx(), GEM_COPAD);
      detid_to_copads_[id].push_back(mydigi);
      superchamber_to_copads_[ superch_id() ].push_back(mydigi);
    }
  }
}


std::set<unsigned int>
GEMDigiMatcher::detIds() const
{
  std::set<unsigned int> result;
  for (auto& p: detid_to_digis_) result.insert(p.first);
  return result;
}


std::set<unsigned int>
GEMDigiMatcher::chamberIds() const
{
  std::set<unsigned int> result;
  for (auto& p: chamber_to_digis_) result.insert(p.first);
  return result;
}

std::set<unsigned int>
GEMDigiMatcher::superChamberIds() const
{
  std::set<unsigned int> result;
  for (auto& p: superchamber_to_digis_) result.insert(p.first);
  return result;
}


std::set<unsigned int>
GEMDigiMatcher::detIdsWithCoPads() const
{
  std::set<unsigned int> result;
  for (auto& p: detid_to_copads_) result.insert(p.first);
  return result;
}

std::set<unsigned int>
GEMDigiMatcher::superChamberIdsWithCoPads() const
{
  std::set<unsigned int> result;
  for (auto& p: superchamber_to_copads_) result.insert(p.first);
  return result;
}




const matching::DigiContainer&
GEMDigiMatcher::digisInDetId(unsigned int detid) const
{
  if (detid_to_digis_.find(detid) == detid_to_digis_.end()) return no_digis_;
  return detid_to_digis_.at(detid);
}

const matching::DigiContainer&
GEMDigiMatcher::digisInChamber(unsigned int detid) const
{
  if (chamber_to_digis_.find(detid) == chamber_to_digis_.end()) return no_digis_;
  return chamber_to_digis_.at(detid);
}

const matching::DigiContainer&
GEMDigiMatcher::digisInSuperChamber(unsigned int detid) const
{
  if (superchamber_to_digis_.find(detid) == superchamber_to_digis_.end()) return no_digis_;
  return superchamber_to_digis_.at(detid);
}

const matching::DigiContainer&
GEMDigiMatcher::padsInDetId(unsigned int detid) const
{
  if (detid_to_pads_.find(detid) == detid_to_pads_.end()) return no_digis_;
  return detid_to_pads_.at(detid);
}

const matching::DigiContainer&
GEMDigiMatcher::padsInChamber(unsigned int detid) const
{
  if (chamber_to_pads_.find(detid) == chamber_to_pads_.end()) return no_digis_;
  return chamber_to_pads_.at(detid);
}

const matching::DigiContainer&
GEMDigiMatcher::padsInSuperChamber(unsigned int detid) const
{
  if (superchamber_to_pads_.find(detid) == superchamber_to_pads_.end()) return no_digis_;
  return superchamber_to_pads_.at(detid);
}


const matching::DigiContainer&
GEMDigiMatcher::coPadsInDetId(unsigned int detid) const
{
  if (detid_to_copads_.find(detid) == detid_to_copads_.end()) return no_digis_;
  return detid_to_copads_.at(detid);
}

const matching::DigiContainer&
GEMDigiMatcher::coPadsInSuperChamber(unsigned int detid) const
{
  if (superchamber_to_copads_.find(detid) == superchamber_to_copads_.end()) return no_digis_;
  return superchamber_to_copads_.at(detid);
}


int
GEMDigiMatcher::nLayersWithDigisInSuperChamber(unsigned int detid) const
{
  set<int> layers;
  auto digis = digisInSuperChamber(detid);
  for (auto& d: digis)
  {
    GEMDetId idd(digi_id(d));
    layers.insert(idd.layer());
  }
  return layers.size();
}


int
GEMDigiMatcher::nLayersWithPadsInSuperChamber(unsigned int detid) const
{
  set<int> layers;
  auto digis = padsInSuperChamber(detid);
  for (auto& d: digis)
  {
    GEMDetId idd(digi_id(d));
    layers.insert(idd.layer());
  }
  return layers.size();
}


int
GEMDigiMatcher::nCoPads() const
{
  int n = 0;
  auto ids = superChamberIdsWithCoPads();
  for (auto id: ids)
  {
    n += coPadsInSuperChamber(id).size();
  }
  return n;
}


int
GEMDigiMatcher::nPads() const
{
  int n = 0;
  auto ids = superChamberIds();
  for (auto id: ids)
  {
    n += padsInSuperChamber(id).size();
  }
  return n;
}


std::set<int>
GEMDigiMatcher::stripNumbersInDetId(unsigned int detid) const
{
  set<int> result;
  auto digis = digisInDetId(detid);
  for (auto& d: digis)
  {
    result.insert( digi_channel(d) );
  }
  return result;
}

std::set<int>
GEMDigiMatcher::padNumbersInDetId(unsigned int detid) const
{
  set<int> result;
  auto digis = padsInDetId(detid);
  for (auto& d: digis)
  {
    result.insert( digi_channel(d) );
  }
  return result;
}

std::set<int>
GEMDigiMatcher::coPadNumbersInDetId(unsigned int detid) const
{
  set<int> result;
  auto digis = coPadsInDetId(detid);
  for (auto& d: digis)
  {
    result.insert( digi_channel(d) );
  }
  return result;
}

std::set<int>
GEMDigiMatcher::partitionNumbers() const
{
  std::set<int> result;

  auto detids = detIds();
  for (auto id: detids)
  {
    GEMDetId idd(id);
    result.insert( idd.roll() );
  }
  return result;
}

std::set<int>
GEMDigiMatcher::partitionNumbersWithCoPads() const
{
  std::set<int> result;

  auto detids = detIdsWithCoPads();
  for (auto id: detids)
  {
    GEMDetId idd(id);
    result.insert( idd.roll() );
  }
  return result;
}

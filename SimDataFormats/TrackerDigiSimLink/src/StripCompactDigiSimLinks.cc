#include "SimDataFormats/TrackerDigiSimLink/interface/StripCompactDigiSimLinks.h"

#include <algorithm>
#include <boost/foreach.hpp>


StripCompactDigiSimLinks::Links 
StripCompactDigiSimLinks::getLinks(const StripCompactDigiSimLinks::key_type &key) const 
{
    std::vector<TrackRecord>::const_iterator last  = trackRecords_.end();
    std::vector<TrackRecord>::const_iterator match = std::lower_bound(trackRecords_.begin(), last, key);
    if ((match != last) && (*match == key)) {
        // std::vector<TrackRecord>::const_iterator next = match+1;
        unsigned int end = (match+1 == last ? hitRecords_.size() : (match+1)->start);
        return Links(hitRecords_.begin()+match->start, hitRecords_.begin()+end);
    } else {
        return Links(hitRecords_.end(), hitRecords_.end());
    }
}

StripCompactDigiSimLinks::StripCompactDigiSimLinks(const StripCompactDigiSimLinks::Filler &filler) 
{
    trackRecords_.reserve(filler.keySize());
    hitRecords_.reserve(filler.dataSize());
    BOOST_FOREACH( const Filler::Map::value_type &pair, filler.storage() ) {
        trackRecords_.push_back(TrackRecord(pair.first, hitRecords_.size()));
        hitRecords_.insert(hitRecords_.end(), pair.second.begin(), pair.second.end());
    }
}

StripCompactDigiSimLinks::Filler::~Filler() {
}

void
StripCompactDigiSimLinks::Filler::insert(const StripCompactDigiSimLinks::key_type &key, const StripCompactDigiSimLinks::HitRecord &record) 
{
    Filler::Map::iterator it = storage_.find(key);
    if (it == storage_.end()) {
        storage_.insert(std::make_pair(key, std::vector<HitRecord>(1,record)));
        num_keys_++; 
        num_values_++;
    } else {
        it->second.push_back(record);
        num_values_++;
    }
}

std::map<uint32_t, std::vector<StripCompactDigiSimLinks::RevLink> > 
StripCompactDigiSimLinks::makeReverseMap() const
{
    std::map<uint32_t, std::vector<StripCompactDigiSimLinks::RevLink> > ret;
    typedef std::vector<TrackRecord>::const_iterator trk_it;
    typedef std::vector<HitRecord>::const_iterator   hit_it;
    hit_it hstart = hitRecords_.begin(), ith = hstart;
    for (trk_it itt = trackRecords_.begin(), endt = trackRecords_.end(); itt != endt; ++itt) {
        hit_it edh = (itt+1 != endt ? hstart + (itt+1)->start : hitRecords_.end());
        for (; ith < edh; ++ith) {
            ret[ith->detId].push_back(RevLink(*itt, *ith));
        }
    }
    return ret;
}


#ifndef SimDataFormats_TrackerDigiSimLink_interface_StripCompactDigiSimLinks_h
#define SimDataFormats_TrackerDigiSimLink_interface_StripCompactDigiSimLinks_h

#include <map>
#include <algorithm>
#include <boost/cstdint.hpp>
#include <boost/range.hpp>
#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"

class StripCompactDigiSimLinks {
    public:
        /// Empty constructor, for ROOT persistence
        StripCompactDigiSimLinks() {}
        ~StripCompactDigiSimLinks() {}

        typedef std::pair<EncodedEventId,unsigned int> key_type;
        struct HitRecord {
            HitRecord() {}
            HitRecord(uint32_t detid, uint16_t first, uint16_t size) : 
                 detId(detid), firstStrip(first), nStrips(size) {}
            uint32_t detId;
            uint16_t firstStrip;
            uint16_t nStrips;
        };
        typedef boost::sub_range<const std::vector<HitRecord> > Links;
        typedef Links value_type;
        
        void swap(StripCompactDigiSimLinks &other) {
            using std::swap;
            swap(trackRecords_, other.trackRecords_);
            swap(hitRecords_,   other.hitRecords_);
        }
       
        Links getLinks(const key_type &key) const ;
        Links operator[](const key_type &key) const { return getLinks(key); }
        
        class Filler {
            public:
                Filler() : storage_(), num_keys_(0), num_values_(0) {}

                // out of line destructor to avoid code bloat
                ~Filler() ;

                void insert(const key_type  &key, const HitRecord &record);

                typedef std::map<key_type, std::vector<HitRecord> >  Map;
                const Map & storage() const { return storage_; }

                unsigned int keySize()  const { return num_keys_;   }
                unsigned int dataSize() const { return num_values_; }
            private:
                Map storage_;
                unsigned int num_keys_;
                unsigned int num_values_;
        };
        /// This is the real constructor you will use
        StripCompactDigiSimLinks(const Filler &filler) ;

        struct TrackRecord {
            TrackRecord(){}
            TrackRecord(key_type k, unsigned int offset) : key(k), start(offset) {}
            key_type       key;
            unsigned int   start;    // first index in HitRecord
            //unsigned int   length;
            inline bool operator< (const TrackRecord &other) const { return key <  other.key; }
            inline bool operator< (const key_type &otherKey) const { return key <  otherKey;  }
            inline bool operator==(const key_type &otherKey) const { return key == otherKey;  }
        };

        struct RevLink {
            RevLink(const TrackRecord &track, const HitRecord &hit) : 
                eventId(track.key.first), simTrackId(track.key.second), 
                firstStrip(hit.firstStrip), lastStrip(hit.firstStrip+hit.nStrips-1) {}
            EncodedEventId eventId;
            unsigned int simTrackId;
            uint16_t firstStrip;
            uint16_t lastStrip;
        };

        /// Make the map in the reverse direction. SLOW! call it only once.
        std::map<uint32_t, std::vector<RevLink> > makeReverseMap() const ;
    private:
        /// These MUST be sorted at the same time by key and by start
        /// The rule is enforced by allowing to create this only through a Filler
        std::vector<TrackRecord> trackRecords_; 

        std::vector<HitRecord>   hitRecords_;
};

inline void swap(StripCompactDigiSimLinks &a, StripCompactDigiSimLinks &b) { a.swap(b); }
#endif

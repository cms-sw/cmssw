/*
 *  TrackQuality.cc
 *
 *  Created by Christophe Saout on 9/25/08.
 *  Copyright 2007 __MyCompanyName__. All rights reserved.
 *
 */

#include <algorithm>
#include <vector>

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "DataFormats/MuonDetId/interface/DTLayerId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"

#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"

#include "SimTracker/TrackHistory/interface/TrackQuality.h"

// #define DEBUG_TRACK_QUALITY

namespace
{
static const uint32_t NonMatchedTrackId = (uint32_t)-1;

struct MatchedHit
{
    DetId detId;
    uint32_t simTrackId;
    EncodedEventId collision;
    int recHitId;
    TrackQuality::Layer::State state;

    bool operator < (const MatchedHit &other) const
    {
        if (detId < other.detId)
            return true;
        else if (detId > other.detId)
            return false;
        else if (collision < other.collision)
            return true;
        else if (other.collision < collision)
            return false;
        else
            return simTrackId < other.simTrackId;
    }

    bool operator == (const MatchedHit &other) const
    {
        return detId == other.detId &&
               collision == other.collision &&
               simTrackId == other.simTrackId;
    }
};

static bool operator < (const MatchedHit &hit, DetId detId)
{
    return hit.detId < detId;
}
static bool operator < (DetId detId, const MatchedHit &hit)
{
    return detId < hit.detId;
}
}

typedef std::pair<TrackQuality::Layer::SubDet, short int> DetLayer;

// in case multiple hits were found, figure out the highest priority
static const int statePriorities[] =
{
    /* Unknown */  3,
    /* Good */     5,
    /* Missed */   0,
    /* Noise */    7,
    /* Bad */      2,
    /* Dead */     4,
    /* Shared */   6,
    /* Misassoc */ 1
};

DetLayer getDetLayer(DetId detId, const TrackerTopology *tTopo)
{
    TrackQuality::Layer::SubDet det = TrackQuality::Layer::Invalid;
    short int layer = 0;

    switch (detId.det())
    {
    case DetId::Tracker:
      layer=tTopo->layer(detId);
      break;

    case DetId::Muon:
        switch (detId.subdetId())
        {
        case MuonSubdetId::DT:
            det = TrackQuality::Layer::MuonDT;
            layer = DTLayerId(detId).layer();
            break;

        case MuonSubdetId::CSC:
            det = TrackQuality::Layer::MuonCSC;
            layer = CSCDetId(detId).layer();
            break;

        case MuonSubdetId::RPC:
            if (RPCDetId(detId).region())
                det = TrackQuality::Layer::MuonRPCEndcap;
            else
                det = TrackQuality::Layer::MuonRPCBarrel;
            layer = RPCDetId(detId).layer();
            break;

        default:
            /* should not get here */
            ;
        }
        break;

    default:
        /* should not get here */
        ;
    }

    return DetLayer(det, layer);
}

TrackQuality::TrackQuality(const edm::ParameterSet &config) :
        associatorPSet_(config.getParameter<edm::ParameterSet>("hitAssociator"))
{
}

void TrackQuality::newEvent(const edm::Event &ev, const edm::EventSetup &es)
{
    associator_.reset(new TrackerHitAssociator(ev, associatorPSet_));
}

void TrackQuality::evaluate(SimParticleTrail const &spt,
                            reco::TrackBaseRef const &tr,
			    const TrackerTopology *tTopo)
{
    std::vector<MatchedHit> matchedHits;

    // iterate over reconstructed hits
    for (trackingRecHit_iterator hit = tr->recHitsBegin();
            hit != tr->recHitsEnd(); ++hit)
    {
        // on which module the hit lies
        DetId detId = (*hit)->geographicalId();

        // FIXME: check for double-sided modules?

        // didn't find a hit on that module
        if (!(*hit)->isValid())
        {
            MatchedHit matchedHit;
            matchedHit.detId = detId;
            matchedHit.simTrackId = NonMatchedTrackId;
            // check why hit wasn't valid and propagate information
            switch ((*hit)->getType())
            {
            case TrackingRecHit::inactive:
                matchedHit.state = Layer::Dead;
                break;

            case TrackingRecHit::bad:
                matchedHit.state = Layer::Bad;
                break;

            default:
                matchedHit.state = Layer::Missed;
            }
            matchedHit.recHitId = hit - tr->recHitsBegin();
            matchedHits.push_back(matchedHit);
            continue;
        }

        // find simulated tracks causing hit that was reconstructed
        std::vector<SimHitIdpr> simIds = associator_->associateHitId(**hit);

        // must be noise or so
        if (simIds.empty())
        {
            MatchedHit matchedHit;
            matchedHit.detId = detId;
            matchedHit.simTrackId = NonMatchedTrackId;
            matchedHit.state = Layer::Noise;
            matchedHit.recHitId = hit - tr->recHitsBegin();
            matchedHits.push_back(matchedHit);
            continue;
        }

        // register all simulated tracks contributing
        for (std::vector<SimHitIdpr>::const_iterator i = simIds.begin();
                i != simIds.end(); ++i)
        {
            MatchedHit matchedHit;
            matchedHit.detId = detId;
            matchedHit.simTrackId = i->first;
            matchedHit.collision = i->second;
            // RecHit <-> SimHit matcher currently doesn't support muon system
            if (detId.det() == DetId::Muon)
                matchedHit.state = Layer::Unknown;
            else
                // assume hit was mismatched (until possible confirmation)
                matchedHit.state = Layer::Misassoc;
            matchedHit.recHitId = hit - tr->recHitsBegin();
            matchedHits.push_back(matchedHit);
        }
    }

    // sort hits found so far by module id
    std::stable_sort(matchedHits.begin(), matchedHits.end());

    std::vector<MatchedHit>::size_type size = matchedHits.size();

    // now iterate over simulated hits and compare (tracks in chain first)
    for (SimParticleTrail::const_iterator track = spt.begin();
            track != spt.end(); ++track)
    {
        // iterate over all hits in track
        for (std::vector<PSimHit>::const_iterator hit =
                    (*track)->pSimHit_begin();
                hit != (*track)->pSimHit_end(); ++hit)
        {
            MatchedHit matchedHit;
            matchedHit.detId = DetId(hit->detUnitId());
            matchedHit.simTrackId = hit->trackId();
            matchedHit.collision = hit->eventId();

            // find range of reconstructed hits belonging to this module
            std::pair<std::vector<MatchedHit>::iterator,
            std::vector<MatchedHit>::iterator>
            range = std::equal_range(
                        matchedHits.begin(),
                        matchedHits.begin() + size,
                        matchedHit.detId);

            // no reconstructed hit found, remember this as a missed module
            if (range.first == range.second)
            {
                matchedHit.state = Layer::Missed;
                matchedHit.recHitId = -1;
                matchedHits.push_back(matchedHit);
                continue;
            }

            // now find if the hit belongs to the correct simulated track
            std::vector<MatchedHit>::iterator pos =
                std::lower_bound(range.first,
                                 range.second,
                                 matchedHit);

            // if it does, check for being a shared hit (was Misassoc before)
            if (pos != range.second)
            {
                if (range.second - range.first > 1) // more than one SimHit
                    pos->state = Layer::Shared;
                else
                    pos->state = Layer::Good; // only hit -> good hit
            }
        }
    }

    // in case we added missed modules, re-sort
    std::stable_sort(matchedHits.begin(), matchedHits.end());

    // prepare for ordering results by layer enum and layer/disk number
    typedef std::multimap<DetLayer, const MatchedHit*> LayerHitMap;
    LayerHitMap layerHitMap;

    // iterate over all simulated/reconstructed hits again
    for (std::vector<MatchedHit>::const_iterator hit = matchedHits.begin();
            hit != matchedHits.end();)
    {
        // we can have multiple reco-to-sim matches per module, find best one
        const MatchedHit *best = 0;

        // this loop iterates over all subsequent hits in the same module
        do
        {
            // update our best hit pointer
            if (!best ||
                    statePriorities[hit->state] > statePriorities[best->state] ||
                    best->simTrackId == NonMatchedTrackId)
            {
                best = &*hit;
            }
            ++hit;
        }
        while (hit != matchedHits.end() &&
                hit->detId == best->detId);

        // ignore hit in case track reco was looking at the wrong module
        if (best->simTrackId != NonMatchedTrackId ||
                best->state != Layer::Missed)
        {
            layerHitMap.insert(std::make_pair(
					      getDetLayer(best->detId,tTopo), best));
        }
    }

    layers_.clear();

#ifdef DEBUG_TRACK_QUALITY
    std::cout << "---------------------" << std::endl;
#endif
    // now prepare final collection
    for (LayerHitMap::const_iterator hit = layerHitMap.begin();
            hit != layerHitMap.end(); ++hit)
    {
#ifdef DEBUG_TRACK_QUALITY
        std::cout
            << "detLayer (" << hit->first.first << ", " << hit->first.second << ")"
            << " [" << (uint32_t)hit->second->detId << "] sim(" << (int)hit->second->simTrackId << ")"
            << " hit(" << hit->second->recHitId << ") -> " << hit->second->state << std::endl;
#endif

        // find out if we need to start a new layer
        Layer *layer = layers_.empty() ? 0 : &layers_.back();
        if (!layer ||
                hit->first.first != layer->subDet ||
                hit->first.second != layer->layer)
        {
            Layer newLayer;
            newLayer.subDet = hit->first.first;
            newLayer.layer = hit->first.second;
            layers_.push_back(newLayer);
            layer = &layers_.back();
        }

        // create hit and add it to layer
        Layer::Hit newHit;
        newHit.recHitId = hit->second->recHitId;
        newHit.state = hit->second->state;

        layer->hits.push_back(newHit);
    }
}

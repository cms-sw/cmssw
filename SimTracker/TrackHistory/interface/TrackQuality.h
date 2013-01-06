/*
 *  TrackQuality.h
 *
 *  Created by Christophe Saout on 9/25/08.
 *  Copyright 2007 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef TrackQuality_h
#define TrackQuality_h

#include <vector>
#include <memory>

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"

class TrackerTopology;

//! This class analyses the reconstruction quality for a given track
class TrackQuality
{
public:
    typedef std::vector<TrackingParticleRef> SimParticleTrail;

    struct Layer
    {
        enum SubDet
        {
            Invalid = 0,
            PixelBarrel, PixelForward,
            StripTIB, StripTID, StripTOB, StripTEC,
            MuonDT, MuonCSC, MuonRPCBarrel, MuonRPCEndcap
        };

        enum State
        {
            Unknown = 0,
            Good,
            Missed,
            Noise,
            Bad,
            Dead,
            Shared,
            Misassoc
        };

        struct Hit
        {
            short int recHitId;
            State state;
        };

        SubDet subDet;
        short int layer;
        std::vector<Hit> hits;
    };

public:
    //! Constructor by pset.
    /* Creates a TrackQuality object from a pset.

       /param[in] pset with the configuration values
    */
    TrackQuality(const edm::ParameterSet &);

    //! Pre-process event information (for accessing reconstruction information)
    void newEvent(const edm::Event &, const edm::EventSetup &);

    //! Compute information about the track reconstruction quality
    void evaluate(SimParticleTrail const &, reco::TrackBaseRef const &, const TrackerTopology *tTopo);

    //! Return the number of layers with simulated and/or reconstructed hits
    unsigned int numberOfLayers() const
    {
        return layers_.size();
    }

    //! Return information about the given layer by index
    const Layer &layer(unsigned int index) const
    {
        return layers_[index];
    }

private:
    const edm::ParameterSet associatorPSet_;
    std::auto_ptr<TrackerHitAssociator> associator_;

    std::vector<Layer> layers_;
};

#endif

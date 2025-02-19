#ifndef SimDataFormats_TrackingDataPrint_h
#define SimDataFormats_TrackingDataPrint_h
/** Concrete TrackingParticle.
 *  All track parameters are passed in the constructor and stored internally.
 */

#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertex.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include <iostream>

std::ostream& operator<< (std::ostream& s, const TrackingParticle & tp)
{

    // Compare momenta from sources
    s << "T.P.   Track Momentum, q , ID, & Event # "
    << tp.p4()    << " " << tp.charge() << " "
    << tp.pdgId() << " "
    << tp.eventId().bunchCrossing() << "." << tp.eventId().event()
    << std::endl;
    s << " Hits for this track: " << tp.trackPSimHit().size()
    << std::endl;

    for (TrackingParticle::genp_iterator hepT = tp.genParticle_begin();
            hepT !=  tp.genParticle_end(); ++hepT)
    {
        s << " HepMC Track Momentum " << (*hepT)->momentum().mag() << std::endl;
    }
    for (TrackingParticle::g4t_iterator g4T = tp.g4Track_begin();
            g4T !=  tp.g4Track_end(); ++g4T)
    {
        s << " Geant Track Momentum  " << g4T->momentum() << std::endl;
        s << " Geant Track ID & type " << g4T->trackId() << " "
        << g4T->type() << std::endl;
        if (g4T->type() !=  tp.pdgId())
        {
            s << " Mismatch b/t TrackingParticle and Geant types"
            << std::endl;
        }
    }
    return s;
}

#endif // SimDataFormats_TrackingDataPrint_H

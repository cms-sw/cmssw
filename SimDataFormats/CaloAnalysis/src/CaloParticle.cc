#include "SimDataFormats/CaloAnalysis/interface/CaloParticle.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include <FWCore/MessageLogger/interface/MessageLogger.h>

const unsigned int CaloParticle::longLivedTag = 65536;

CaloParticle::CaloParticle()
{
  // No operation
}

CaloParticle::CaloParticle( const SimTrack& simtrk )
{
  addG4Track( simtrk );
  event_ = simtrk.eventId();
  particleId_ = simtrk.trackId();
}

CaloParticle::CaloParticle( EncodedEventId eventID, uint32_t particleID )
{
  event_ = eventID;
  particleId_ = particleID;
}

CaloParticle::~CaloParticle()
{
}

std::ostream& operator<< (std::ostream& s, CaloParticle const & tp)
{
    s << "CP momentum, q, ID, & Event #: "
    << tp.p4()                      << " " << tp.charge() << " "   << tp.pdgId() << " "
    << tp.eventId().bunchCrossing() << "." << tp.eventId().event() << std::endl;

    for (CaloParticle::genp_iterator hepT = tp.genParticle_begin(); hepT !=  tp.genParticle_end(); ++hepT)
    {
        s << " HepMC Track Momentum " << (*hepT)->momentum().rho() << std::endl;
    }

    for (CaloParticle::g4t_iterator g4T = tp.g4Track_begin(); g4T !=  tp.g4Track_end(); ++g4T)
    {
        s << " Geant Track Momentum  " << g4T->momentum() << std::endl;
        s << " Geant Track ID & type " << g4T->trackId() << " " << g4T->type() << std::endl;
        if (g4T->type() !=  tp.pdgId())
        {
            s << " Mismatch b/t CaloParticle and Geant types" << std::endl;
        }
    }
    return s;
}

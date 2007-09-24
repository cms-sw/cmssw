/*
 *  TrackOrigin.C
 *  CMSSW_1_3_1
 *
 *  Created by Victor Eduardo Bazterra on 5/30/07.
 *  Copyright 2007 __MyCompanyName__. All rights reserved.
 *
 */

#include <math.h>

#include "SimTracker/TrackHistory/interface/TrackOrigin.h"	

#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertexContainer.h"

bool TrackOrigin::hasParticles(std::set<int> const & list) const
{
  // look into the TrackingParticle trail
  TrackingParticleRefVector::const_iterator tpi;
  for (tpi = simParticleTrail_.begin(); tpi != simParticleTrail_.end(); tpi++)
     if (list.find((*tpi)->pdgId()) != list.end())
       return true;

  // look into the GenParticle trail
  GenParticleTrail::const_iterator gpi;
  for (gpi = genParticleTrail_.begin(); gpi != genParticleTrail_.end(); gpi++)
    if (list.find((*gpi)->pdg_id()) != list.end())
      return true;
      
  return false;
}


bool TrackOrigin::hasPhotonConversion() const
{
  TrackingVertexContainer::const_iterator tvr;
  TrackingParticleRefVector sources, daughters;
  for (tvr = simVertexTrail_.begin(); tvr != simVertexTrail_.end(); tvr++)
  {
    sources = (*tvr)->sourceTracks();
    daughters = (*tvr)->daughterTracks();
    if (sources.size() == 1              &&  // require one source 
        daughters.size() == 2            &&  //    "    two daughters
        sources[0]->pdgId() == 22        &&  //    "    a photon in the source
        abs(daughters[0]->pdgId()) == 11 &&  //    "    two electrons
        abs(daughters[1]->pdgId()) == 11
    ) return true;
  }
  return false;
}

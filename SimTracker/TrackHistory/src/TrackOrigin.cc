/*
 *  TrackOrigin.C
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
  SimParticleTrail::const_iterator tpi;
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


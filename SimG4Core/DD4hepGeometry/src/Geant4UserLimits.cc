//==========================================================================
//  AIDA Detector description implementation 
//--------------------------------------------------------------------------
// Copyright (C) Organisation europeenne pour la Recherche nucleaire (CERN)
// All rights reserved.
//
// For the licensing terms see $DD4hepINSTALL/LICENSE.
// For the list of contributors see $DD4hepINSTALL/doc/CREDITS.
//
// Author     : M.Frank
//
//==========================================================================

// Framework include files
//FIXME: #include "DDG4/Geant4UserLimits.h"
#include "SimG4Core/DD4hepGeometry/interface/Geant4UserLimits.h"
//FIXME: #include "DDG4/Geant4Particle.h"
#include "SimG4Core/DD4hepGeometry/interface/Geant4Particle.h"
#include "DD4hep/InstanceCount.h"
#include "DD4hep/DD4hepUnits.h"
#include "DD4hep/Primitives.h"

// Geant 4 include files
#include "G4Track.hh"
#include "CLHEP/Units/SystemOfUnits.h"

// C/C++ include files
#include <stdexcept>

using namespace std;
using namespace dd4hep::sim;

/// Access value according to track
double Geant4UserLimits::Handler::value(const G4Track& track) const    {
  if ( !particleLimits.empty() )  {
    auto i = particleLimits.find(track.GetDefinition());
    if ( i != particleLimits.end() )  {
      return (*i).second;
    }
  }
  return defaultValue;
}

/// Set the handler value(s)
void Geant4UserLimits::Handler::set(const string& particles, double val)   {
  if ( particles == "*" )   {
    defaultValue = val;
    return;
  }
  auto defs = Geant4ParticleHandle::g4DefinitionsRegEx(particles);
  for(auto* d : defs)
    particleLimits[d] = val;
}

/// Initializing Constructor
Geant4UserLimits::Geant4UserLimits(LimitSet ls)
  : G4UserLimits(ls.name()), limits(ls)
{
  const auto& lim = limits.limits();
  InstanceCount::increment(this);
  /// Set defaults
  maxStepLength.defaultValue  = fMaxStep;
  maxTrackLength.defaultValue = fMaxTrack;
  maxTime.defaultValue        = fMaxTime;
  minEKine.defaultValue       = fMinEkine;
  minRange.defaultValue       = fMinRange;
  /// Overwrite with values if present:
  for(const Limit& l : lim)   {
    if (l.name == "step_length_max")
      maxStepLength.set(l.particles, l.value*CLHEP::mm/dd4hep::mm);
    else if (l.name == "track_length_max")
      maxTrackLength.set(l.particles, l.value*CLHEP::mm/dd4hep::mm);
    else if (l.name == "time_max")
      maxTime.set(l.particles, l.value*CLHEP::ns/dd4hep::ns);
    else if (l.name == "ekin_min")
      minEKine.set(l.particles, l.value*CLHEP::MeV/dd4hep::MeV);
    else if (l.name == "range_min")
      minRange.set(l.particles, l.value);
    else
      throw runtime_error("Unknown Geant4 user limit: " + l.toString());
  }
}

/// Standard destructor
Geant4UserLimits::~Geant4UserLimits()  {
  InstanceCount::decrement(this);
}

/// Setters may not be called!
void Geant4UserLimits::SetMaxAllowedStep(G4double /* ustepMax */)  {
  dd4hep::notImplemented(string(__PRETTY_FUNCTION__)+" May not be called!");
}

void Geant4UserLimits::SetUserMaxTrackLength(G4double /* utrakMax */)  {
  dd4hep::notImplemented(string(__PRETTY_FUNCTION__)+" May not be called!");
}

void Geant4UserLimits::SetUserMaxTime(G4double /* utimeMax */)  {
  dd4hep::notImplemented(string(__PRETTY_FUNCTION__)+" May not be called!");
}

void Geant4UserLimits::SetUserMinEkine(G4double /* uekinMin */)  {
  dd4hep::notImplemented(string(__PRETTY_FUNCTION__)+" May not be called!");
}

void Geant4UserLimits::SetUserMinRange(G4double /* urangMin */)  {
  dd4hep::notImplemented(string(__PRETTY_FUNCTION__)+" May not be called!");
}


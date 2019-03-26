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
#ifndef DD4HEP_DDG4_GEANT4USERLIMITS_H
#define DD4HEP_DDG4_GEANT4USERLIMITS_H

// Framework include files
#include "DD4hep/Objects.h"

// Geant 4 include files
#include "G4UserLimits.hh"

// Forward declarations
class G4ParticleDefinition;

/// Namespace for the AIDA detector description toolkit
namespace dd4hep {


  /// Namespace for the Geant4 based simulation part of the AIDA detector description toolkit
  namespace sim {

    /// Helper to dump Geant4 volume hierarchy
    /**
     *  \author  M.Frank
     *  \version 1.0
     *  \ingroup DD4HEP_SIMULATION
     */
    class Geant4UserLimits : public  G4UserLimits {
    public:
      /// Helper class to one limit type
      /**
       *  \author  M.Frank
       *  \version 1.0
       *  \ingroup DD4HEP_SIMULATION
       */
      struct Handler  {
      public:
        /// Default value (either from base class or value if Limit.particles='*')
        double                defaultValue = 0.0;
        /// Handler particle ids for the limit (pdgID)
        std::map<const G4ParticleDefinition*, double> particleLimits;
      public:
        /// Default constructor
        Handler() = default;
        /// Set the handler value(s)
        void set(const std::string& particles, double val);
        /// Access value according to track
        double value(const G4Track& track) const;
      };
      /// Handle to the limitset to be applied.
      LimitSet  limits;
      /// Handler map for MaxStepLength limit
      Handler   maxStepLength;
      /// Handler map for MaxTrackLength limit
      Handler   maxTrackLength;
      /// Handler map for MaxTime limit
      Handler   maxTime;
      /// Handler map for MinEKine limit
      Handler   minEKine;
      /// Handler map for MinRange limit
      Handler   minRange;

    public:
      /// Initializing Constructor
      Geant4UserLimits(LimitSet ls);
      /// Standard destructor
      virtual ~Geant4UserLimits();
      /// Access the user tracklength for a G4 track object
      virtual G4double GetMaxAllowedStep(const G4Track& track)
      {  return maxStepLength.value(track);    }
      /// Access the user tracklength for a G4 track object
      virtual G4double GetUserMaxTrackLength(const G4Track& track)
      {  return maxTrackLength.value(track);   }
      /// Access the proper time cut for a G4 track object
      virtual G4double GetUserMaxTime (const G4Track& track)
      {  return maxTime.value(track);          }
      /// Access the kinetic energy cut for a G4 track object
      virtual G4double GetUserMinEkine(const G4Track& track)
      {  return minEKine.value(track);         }
      /// Access the range cut for a G4 track object
      virtual G4double GetUserMinRange(const G4Track& track)
      {  return minRange.value(track);         }
      /// Setters may not be called!
      virtual void SetMaxAllowedStep(G4double ustepMax);    
      virtual void SetUserMaxTrackLength(G4double utrakMax);
      virtual void SetUserMaxTime(G4double utimeMax);
      virtual void SetUserMinEkine(G4double uekinMin);
      virtual void SetUserMinRange(G4double urangMin);
    };
  }
}

#endif  // DD4HEP_DDG4_GEANT4USERLIMITS_H

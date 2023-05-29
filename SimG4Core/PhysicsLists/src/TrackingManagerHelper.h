//
// ********************************************************************
// * License and Disclaimer                                           *
// *                                                                  *
// * The  Geant4 software  is  copyright of the Copyright Holders  of *
// * the Geant4 Collaboration.  It is provided  under  the terms  and *
// * conditions of the Geant4 Software License,  included in the file *
// * LICENSE and available at  http://cern.ch/geant4/license .  These *
// * include a list of copyright holders.                             *
// *                                                                  *
// * Neither the authors of this software system, nor their employing *
// * institutes,nor the agencies providing financial support for this *
// * work  make  any representation or  warranty, express or implied, *
// * regarding  this  software system or assume any liability for its *
// * use.  Please see the license in the file  LICENSE  and URL above *
// * for the full disclaimer and the limitation of liability.         *
// *                                                                  *
// * This  code  implementation is the result of  the  scientific and *
// * technical work of the GEANT4 collaboration.                      *
// * By using,  copying,  modifying or  distributing the software (or *
// * any work based  on the software)  you  agree  to acknowledge its *
// * use  in  resulting  scientific  publications,  and indicate your *
// * acceptance of all terms of the Geant4 Software license.          *
// ********************************************************************
//
// TrackingManagerHelper
//
// Class description:
//
// Helper class for reducing the effort required to implement a custom tracking
// manager. It implements a stepping loop that calls user actions as the generic
// tracking and stepping managers do, and it implements navigation for charged
// particles in energy-preserving fields and for neutral particles.
//
// Original author: Jonas Hahnfeld, 2021

#ifndef TrackingManagerHelper_hh
#define TrackingManagerHelper_hh 1

#include "G4TrackVector.hh"
#include "globals.hh"

class G4Step;
class G4Track;

class TrackingManagerHelper {
public:
  class Physics {
  public:
    virtual ~Physics() {}

    virtual void StartTracking(G4Track*) {}
    virtual void EndTracking() {}

    // Combines AlongStep and PostStep; the implementation needs to remember
    // the right value to pass as previousStepSize to G4VProcess.
    virtual G4double GetPhysicalInteractionLength(const G4Track& track) = 0;

    // This method is called for every step after navigation. The updated
    // position is stored in the G4Step's post-step point. Any particle change
    // should be applied directly to the step, UpdateTrack() will be called
    // automatically after this method returns. If secondaries should be given
    // back to the G4EventManager, put them into the container passed as the
    // last argument.
    virtual void AlongStepDoIt(G4Track& track, G4Step& step, G4TrackVector& secondaries) = 0;

    // This method is called unless the track has been killed during this step.
    // If secondaries should be given back to the G4EventManager, put them into
    // the container passed as the last argument.
    virtual void PostStepDoIt(G4Track& track, G4Step& step, G4TrackVector& secondaries) = 0;

    virtual bool HasAtRestProcesses() { return false; }

    // This method is called when a track is stopped, but still alive. If
    // secondaries should be given back to the G4EventManager, put them into
    // the container passed as the last argument.
    virtual void AtRestDoIt(G4Track& track, G4Step& step, G4TrackVector& secondaries) {
      (void)track;
      (void)step;
      (void)secondaries;
    }
  };

  class Navigation {
  public:
    virtual ~Navigation() {}

    virtual G4double MakeStep(G4Track& track, G4Step& step, G4double physicalStep) = 0;

    virtual void FinishStep(G4Track& track, G4Step& step) = 0;
  };

  template <typename PhysicsImpl, typename NavigationImpl>
  static void TrackParticle(G4Track* aTrack, PhysicsImpl& physics, NavigationImpl& navigation);

  template <typename PhysicsImpl>
  static void TrackChargedParticle(G4Track* aTrack, PhysicsImpl& physics);

  template <typename PhysicsImpl>
  static void TrackNeutralParticle(G4Track* aTrack, PhysicsImpl& physics);
};

#include "TrackingManagerHelper.icc"

#endif

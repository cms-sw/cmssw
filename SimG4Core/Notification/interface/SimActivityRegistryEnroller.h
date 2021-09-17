#ifndef Notification_SimActivityRegistryEnroller_h
#define Notification_SimActivityRegistryEnroller_h
// -*- C++ -*-
//
// Package:     Notification
// Class  :     SimActivityRegistryEnroller
//
/**\class SimActivityRegistryEnroller SimActivityRegistryEnroller.h SimG4Core/Notification/interface/SimActivityRegistryEnroller.h

 Description: Based on what classes an object inherts, enrolls that object with the proper signal

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Sun Nov 13 15:08:12 EST 2005
//
// user include files
#include "SimG4Core/Notification/interface/Observer.h"
#include "SimG4Core/Notification/interface/SimActivityRegistry.h"

// forward declarations
namespace enroller_helper {
  template <class T>
  struct Enrollment {
    static void enroll(SimActivityRegistry& iReg, Observer<const T*>* iObs) { iReg.connect(iObs); }
    static void enroll(SimActivityRegistry&, void*) {}
  };

  template <class T>
  static void enroll(SimActivityRegistry& iReg, T* iT) {}

  template <class T, class F, class... TVector>
  static void enroll(SimActivityRegistry& iReg, T* iT) {
    //Try to enroll the object if it inherits from the class at the
    // start of TVector
    Enrollment<F>::enroll(iReg, iT);
    enroll<T, TVector...>(iReg, iT);
  }

}  // namespace enroller_helper

class SimActivityRegistryEnroller {
public:
  SimActivityRegistryEnroller() {}

  template <class T>
  static void enroll(SimActivityRegistry& iReg, T* iObj) {
    enroller_helper::enroll<T,
                            BeginOfJob,
                            DDDWorld,
                            BeginOfRun,
                            BeginOfEvent,
                            BeginOfTrack,
                            BeginOfStep,
                            G4Step,
                            EndOfTrack,
                            EndOfEvent,
                            EndOfRun>(iReg, iObj);
  }

  // stop default
  SimActivityRegistryEnroller(const SimActivityRegistryEnroller&) = delete;
  const SimActivityRegistryEnroller& operator=(const SimActivityRegistryEnroller&) = delete;
};

#endif

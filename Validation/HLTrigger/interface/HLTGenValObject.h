#ifndef Validation_HLTrigger_HLTGenValObject_h
#define Validation_HLTrigger_HLTGenValObject_h

//********************************************************************************
//
// Description:
//   This class is an object wrapper for the Generator level validation code.
//   It handles the different type of objects the code needs to run on: GenParticles, GenJets and event-level energy sums
//
// Author : Finn Labe, UHH, Nov. 2021
//
//********************************************************************************

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"

#include "DataFormats/Math/interface/LorentzVector.h"
#include <vector>

class HLTGenValObject {
public:
  // empty constructor
  HLTGenValObject() : trigObject_(0, 0., 0., 0., 0.) {}

  // constructor from GenParticle
  HLTGenValObject(const reco::GenParticle &p)
      : p4Polar_(p.p4()), p4Cartesian_(p.p4()), trigObject_(0, 0., 0., 0., 0.) {}

  // constructor from GenJet
  HLTGenValObject(const reco::GenJet &p) : p4Polar_(p.p4()), p4Cartesian_(p.p4()), trigObject_(0, 0., 0., 0., 0.) {}

  // constructor from LorentzVector (for energy sums)
  HLTGenValObject(const reco::Candidate::PolarLorentzVector &p)
      : p4Polar_(p), p4Cartesian_(p), trigObject_(0, 0., 0., 0., 0.) {}

  // object functions, for usage of HLTGenValObjects by other modules
  double pt() const { return p4Polar_.pt(); }
  double eta() const { return p4Polar_.eta(); }
  double phi() const { return p4Polar_.phi(); }
  double et() const { return (pt() <= 0) ? 0 : p4Cartesian_.Et(); }
  double mass() const { return p4Cartesian_.mass(); }

  double hasTrigObject() const { return trigObject_.pt() > 0; }
  const trigger::TriggerObject &trigObject() const { return trigObject_; }
  void setTrigObject(const trigger::TriggerObject &trigObject) { trigObject_ = trigObject; }
  double ptRes() const { return pt() > 0 ? trigObject_.pt() / pt() : 0.; }
  double etaRes() const { return eta() > 0 ? trigObject_.eta() / eta() : 0.; }
  double phiRes() const { return phi() > 0 ? trigObject_.phi() / phi() : 0.; }
  double massRes() const { return mass() > 0 ? trigObject_.mass() / mass() : 0.; }

private:
  // containing information in two "shapes"
  math::PtEtaPhiMLorentzVector p4Polar_;
  math::XYZTLorentzVector p4Cartesian_;
  trigger::TriggerObject trigObject_;
};

#endif

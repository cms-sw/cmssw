
#include "TMTrackTrigger/VertexFinder/interface/TP.h"


#include "SimTracker/Common/interface/TrackingParticleSelector.h"

#include "TMTrackTrigger/VertexFinder/interface/Settings.h"
#include "TMTrackTrigger/VertexFinder/interface/Stub.h"


namespace vertexFinder {

TP::TP(TrackingParticlePtr tpPtr, unsigned int index_in_vTPs, const Settings* settings) : 
  TrackingParticlePtr(tpPtr),
  index_in_vTPs_(index_in_vTPs),
  settings_(settings),
  charge_(tpPtr->charge()),
  mass_(tpPtr->mass()),
  pt_(tpPtr->pt()),
  eta_(tpPtr->eta()),
  theta_(tpPtr->theta()),
  tanLambda_(1./tan(theta_)),
  phi0_(tpPtr->phi()),
  vx_(tpPtr->vertex().x()),
  vy_(tpPtr->vertex().y()),
  vz_(tpPtr->vertex().z()),
  d0_(vx_*sin(phi0_) - vy_*cos(phi0_)), // Copied from CMSSW class TrackBase::d0().
  z0_(vz_ - (vx_*cos(phi0_) + vy_*sin(phi0_))*tanLambda_) // Copied from CMSSW class TrackBase::dz().
{
  const vector<SimTrack> &vst = tpPtr->g4Tracks();
  EncodedEventId eid = vst.at(0).eventId(); 
  inTimeBx_ = (eid.bunchCrossing() == 0); // TP from in-time or out-of-time Bx.
  physicsCollision_ = (eid.event() == 0);  // TP from physics collision or from pileup.

  this->fillUse(); // Fill use_ flag, indicating if TP is worth keeping.
  this->fillUseForEff(); // Fill useForEff_ flag, indicating if TP is good for tracking efficiency measurement.
}


//=== Fill truth info with association from tracking particle to stubs.

void TP::fillTruth(const vector<Stub>& vStubs) {

  for (const Stub& s : vStubs) {
    for (const TP* tp_i : s.assocTPs()) {
      if (tp_i -> index() == this->index()) assocStubs_.push_back(&s);
    }
  }

  this->fillUseForAlgEff(); // Fill useForAlgEff_ flag.

  this->calcNumLayers(); // Calculate number of tracker layers this TP has stubs in.
}


//=== Check if this tracking particle is worth keeping.

void TP::fillUse() {

  const bool useOnlyInTimeParticles = false;
  const bool useOnlyTPfromPhysicsCollisionFalse = false;
  // Use looser cuts here those those used for tracking efficiency measurement.
  // Keep only those TP that have a chance (allowing for finite track resolution) of being reconstructed as L1 tracks. L1 tracks not matching these TP will be defined as fake.

  const vector<int> genPdgIdsAllUnsigned = {11, 13, 211, 321, 2212}; // Include all possible particle types here, as if some are left out, L1 tracks matching one of missing types will be declared fake.
  vector<int> genPdgIdsAll;
  for (unsigned int i = 0; i < genPdgIdsAllUnsigned.size(); i++) {
    genPdgIdsAll.push_back(  genPdgIdsAllUnsigned[i] );
    genPdgIdsAll.push_back( -genPdgIdsAllUnsigned[i] );
  }

  static TrackingParticleSelector trackingParticleSelector( 
							   min(2.0, settings_->genMinPt()),
							 - max(3.5, settings_->genMaxAbsEta()),
							   max(3.5, settings_->genMaxAbsEta()),
							   max(10.0, settings_->genMaxVertR()),
							   max(35.0, settings_->genMaxVertZ()),
							   0,
							   useOnlyTPfromPhysicsCollisionFalse,
                 useOnlyInTimeParticles,
							   true,
    						 false,
    						 genPdgIdsAll);

  const TrackingParticlePtr tp_ptr(*this); // cast to base class.
  use_ = trackingParticleSelector(*tp_ptr);
}


//=== Check if this tracking particle can be used to measure the L1 tracking efficiency.

void TP::fillUseForEff() {

  useForEff_ = false;
  if (use_) {
    const bool useOnlyInTimeParticles = true;
    const bool useOnlyTPfromPhysicsCollision = true;
    static TrackingParticleSelector trackingParticleSelector( 
                   settings_->genMinPt(),
                  -settings_->genMaxAbsEta(),
                   settings_->genMaxAbsEta(),
                   settings_->genMaxVertR(),
                   settings_->genMaxVertZ(),
                   0,
                   useOnlyTPfromPhysicsCollision,
                   useOnlyInTimeParticles,
                   true,
                   false,
                   settings_->genPdgIds());

    const TrackingParticlePtr tp_ptr(*this); // cast to base class.
    useForEff_ = trackingParticleSelector(*tp_ptr);
  }
}

//=== Check if this tracking particle can be used to measure the L1 tracking algorithmic efficiency (makes stubs in enough layers).

void TP::fillUseForAlgEff() {
  useForAlgEff_ = false;
  if (useForEff_) {
    useForAlgEff_ = (utility::countLayers(settings_, assocStubs_, true) >= settings_->genMinStubLayers());
  } 
}

}

/*
 *  TrackCategories.h
 *  CMSSW_1_3_1
 *
 *  Created by Victor Eduardo Bazterra on 5/29/07.
 *  Copyright 2007 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef TrackCategories_h
#define TrackCategories_h

#include <set>

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"#include "MagneticField/Engine/interface/MagneticField.h"

#include "SimTracker/TrackHistory/interface/TrackOrigin.h"

#include "TrackingTools/PatternTools/interface/TSCPBuilderNoMaterial.h"#include "TrackingTools/Records/interface/TransientTrackRecord.h"#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"


//! Get track history and classify it in function of their .
class TrackCategories {

public:
  
  enum Category {
    Fake = 0,
    Bad,
    SignalEvent,
    PV,
    SV,
    TV,
    Displaced,
    Ks,
    Lambda,
    PhotonConversion,
    Up,
    Down,
    Strange,
    Charm,
    Bottom,
    Light,
    Unknown
  };

  typedef std::vector<bool> Flags;

  //! Constructor by ParameterSet
  TrackCategories( const edm::ParameterSet & pset) : tracer_(pset)
  {
    // Initialize flags	
    reset();

    // Set the history depth after hadronization
    tracer_.depth(-2);
  }

  //! Pre-process event information (for accessing reconstraction information)
  void newEvent(const edm::Event &, const edm::EventSetup &);
  
  //! Classify the RecoTrack in categories.
  bool evaluate (edm::RefToBase<reco::Track>);

  //! Classify the TrackingParticle in categories.
  bool evaluate (TrackingParticleRef);

  //! Classify the RecoTrack in categories (TODO: Remove).
  bool evaluate (reco::TrackRef track)
  { 
  	return evaluate( edm::RefToBase<reco::Track>(track) );
  }

  //! Returns track flag for a given category.
  bool is(Category category) const
  {
    return flags_[category];
  }

  //! Returns track flags with the categories description.
  const Flags & flags() const
  {
    return flags_;
  }

  //! Returns a reference to the track history used in the classification.
  const TrackHistory & history() const
  {
    return tracer_;
  }

private:

  Flags flags_;
  
  TrackOrigin tracer_;

  edm::ESHandle<MagneticField> magneticField_;

  edm::ESHandle<TransientTrackBuilder> transientTrackBuilder_;

  //! Reset the categories flags.
  void reset()
  {
    flags_ = Flags(Unknown + 1, false);
  }

  //! Define all the categories related to track history.
  void byHistory();

  //! Check for long lived particles.
  bool hasLongLived(int) const;
  
  //! Check for phton conversion
  bool hasPhotonConversion() const;

  //! Define all the categories related to reconstruction.
  void byReco(edm::RefToBase<reco::Track>);

};

#endif

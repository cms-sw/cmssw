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
#include <string>

#include "FWCore/Framework/interface/Event.h"#include "FWCore/Framework/interface/ESHandle.h"#include "FWCore/Framework/interface/EventSetup.h"#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"#include "MagneticField/Engine/interface/MagneticField.h"

#include "SimTracker/TrackAssociation/interface/TrackAssociatorBase.h"
#include "SimTracker/TrackHistory/interface/TrackOrigin.h"

#include "TrackingTools/PatternTools/interface/TSCPBuilderNoMaterial.h"#include "TrackingTools/Records/interface/TransientTrackRecord.h"#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"


//! Get track history and classify it in function of their .
class TrackCategories {

public:
  
  enum Category {
    Fake = 0,
    Bad,
    Displaced,
    Up,
    Down,
    Strange,
    Charm,
    Bottom,
    Light,
    Unknown
  };

  typedef std::vector<bool> Flags;

  //! Void constructor
  TrackCategories()
  {
  	// Initialize flags	
    reset();

    // Set the history depth after hadronization
    tracer_.depth(-2);
  }

  //! Constructor by ParameterSet
  TrackCategories( const edm::ParameterSet& );

  //! Pre-process event information (for accessing reconstraction information)
  void event(const edm::Event &, const edm::EventSetup &);
  
  //! classify the RecoTrack in categories.
  bool evaluate (edm::RefToBase<reco::Track>);

  //! classify the TrackingParticle in categories.
  bool evaluate (TrackingParticleRef);

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

  bool associationByHits_;

  std::string trackCollection_;

  reco::RecoToSimCollection association_;

  edm::ESHandle<MagneticField> magneticField_;

  edm::ESHandle<TransientTrackBuilder> transientTrackBuilder_;

  //! Reset the categories flags.
  void reset()
  {
    flags_ = Flags(Unknown + 1, false);
  }

  //! Define all the categories related to track history.
  void byHistory();

  //! Define all the categories related to reconstruction.
  void byReco(edm::RefToBase<reco::Track>);

};

#endif

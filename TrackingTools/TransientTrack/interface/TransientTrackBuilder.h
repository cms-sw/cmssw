#ifndef TRACKINGTOOLS_TRANSIENTRACKBUILDER_H
#define TRACKINGTOOLS_TRANSIENTRACKBUILDER_H

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "FWCore/Framework/interface/Handle.h"

  /**
   * Helper class to build TransientTrack from the persistent Track.
   * This is obtained from the eventSetup, as given in the example in the test
   * directory.
   */

class TransientTrackBuilder {
 public:
    TransientTrackBuilder(const MagneticField* field) :
   	theField(field) {}

    reco::TransientTrack * build ( const reco::Track * p)  const;

    reco::TransientTrack * build ( const reco::TrackRef * p)  const;

    std::vector<reco::TransientTrack> build ( const edm::Handle<reco::TrackCollection> & trkColl)  const;

    const MagneticField* field() const {return theField;}

  private:
    const MagneticField* theField;
};


#endif

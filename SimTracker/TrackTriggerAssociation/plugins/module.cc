/*! \author Nicola Pozzobon
 *  \date   2013, Jul 19
 *
 */

#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"

/// The Associators

#include "SimTracker/TrackTriggerAssociation/plugins/TTClusterAssociator.h"
typedef TTClusterAssociator<Ref_Phase2TrackerDigi_> TTClusterAssociator_Phase2TrackerDigi_;
DEFINE_FWK_MODULE(TTClusterAssociator_Phase2TrackerDigi_);

#include "SimTracker/TrackTriggerAssociation/plugins/TTStubAssociator.h"
typedef TTStubAssociator<Ref_Phase2TrackerDigi_> TTStubAssociator_Phase2TrackerDigi_;
DEFINE_FWK_MODULE(TTStubAssociator_Phase2TrackerDigi_);

#include "SimTracker/TrackTriggerAssociation/plugins/TTTrackAssociator.h"
typedef TTTrackAssociator<Ref_Phase2TrackerDigi_> TTTrackAssociator_Phase2TrackerDigi_;
DEFINE_FWK_MODULE(TTTrackAssociator_Phase2TrackerDigi_);

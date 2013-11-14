/*! \author Nicola Pozzobon
 *  \date   2013, Jul 19
 *
 */

#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"

/// The Associators

#include "SimTracker/TrackTriggerAssociation/plugins/TTClusterAssociator.h"
typedef TTClusterAssociator< Ref_PixelDigi_> TTClusterAssociator_PixelDigi_;
DEFINE_FWK_MODULE( TTClusterAssociator_PixelDigi_ );

#include "SimTracker/TrackTriggerAssociation/plugins/TTStubAssociator.h"
typedef TTStubAssociator< Ref_PixelDigi_ > TTStubAssociator_PixelDigi_;
DEFINE_FWK_MODULE( TTStubAssociator_PixelDigi_ );

#include "SimTracker/TrackTriggerAssociation/plugins/TTTrackAssociator.h"
typedef TTTrackAssociator< Ref_PixelDigi_ > TTTrackAssociator_PixelDigi_;
DEFINE_FWK_MODULE( TTTrackAssociator_PixelDigi_ );


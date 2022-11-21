#ifndef SimTracker_TrackTriggerAssociation_TTTypes_h
#define SimTracker_TrackTriggerAssociation_TTTypes_h

#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTStubAssociationMap.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTClusterAssociationMap.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTTrackAssociationMap.h"

typedef edm::Ptr<TrackingParticle> TPPtr;
typedef TTStubAssociationMap<Ref_Phase2TrackerDigi_> TTStubAssMap;
typedef TTClusterAssociationMap<Ref_Phase2TrackerDigi_> TTClusterAssMap;
typedef TTTrackAssociationMap<Ref_Phase2TrackerDigi_> TTTrackAssMap;

#endif
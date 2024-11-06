#ifndef SimDataFormats_Associations_TTTypes_h
#define SimDataFormats_Associations_TTTypes_h

#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/Associations/interface/TTStubAssociationMap.h"
#include "SimDataFormats/Associations/interface/TTClusterAssociationMap.h"
#include "SimDataFormats/Associations/interface/TTTrackAssociationMap.h"

typedef edm::Ptr<TrackingParticle> TPPtr;
typedef TTStubAssociationMap<Ref_Phase2TrackerDigi_> TTStubAssMap;
typedef TTClusterAssociationMap<Ref_Phase2TrackerDigi_> TTClusterAssMap;
typedef TTTrackAssociationMap<Ref_Phase2TrackerDigi_> TTTrackAssMap;

#endif

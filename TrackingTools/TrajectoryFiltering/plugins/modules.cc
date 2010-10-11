#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "TrackingTools/TrajectoryFiltering/interface/TrajectoryFilterFactory.h"

#include "TrackingTools/TrajectoryFiltering/interface/MaxHitsTrajectoryFilter.h"
#include "TrackingTools/TrajectoryFiltering/interface/MinHitsTrajectoryFilter.h"
#include "TrackingTools/TrajectoryFiltering/interface/MaxLostHitsTrajectoryFilter.h"
#include "TrackingTools/TrajectoryFiltering/interface/MaxConsecLostHitsTrajectoryFilter.h"
#include "TrackingTools/TrajectoryFiltering/interface/MinPtTrajectoryFilter.h"
#include "TrackingTools/TrajectoryFiltering/interface/ThresholdPtTrajectoryFilter.h"
#include "TrackingTools/TrajectoryFiltering/interface/CompositeTrajectoryFilter.h"
#include "TrackingTools/TrajectoryFiltering/interface/CkfBaseTrajectoryFilter.h"
#include "TrackingTools/TrajectoryFiltering/interface/ChargeSignificanceTrajectoryFilter.h"

DEFINE_EDM_PLUGIN(TrajectoryFilterFactory, MaxHitsTrajectoryFilter, "MaxHitsTrajectoryFilter");
DEFINE_EDM_PLUGIN(TrajectoryFilterFactory, MinHitsTrajectoryFilter, "MinHitsTrajectoryFilter");
DEFINE_EDM_PLUGIN(TrajectoryFilterFactory, MaxLostHitsTrajectoryFilter, "MaxLostHitsTrajectoryFilter");
DEFINE_EDM_PLUGIN(TrajectoryFilterFactory, MaxConsecLostHitsTrajectoryFilter, "MaxConsecLostHitsTrajectoryFilter");
DEFINE_EDM_PLUGIN(TrajectoryFilterFactory, MinPtTrajectoryFilter, "MinPtTrajectoryFilter");
DEFINE_EDM_PLUGIN(TrajectoryFilterFactory, ThresholdPtTrajectoryFilter, "ThresholdPtTrajectoryFilter");
DEFINE_EDM_PLUGIN(TrajectoryFilterFactory, CompositeTrajectoryFilter, "CompositeTrajectoryFilter");
DEFINE_EDM_PLUGIN(TrajectoryFilterFactory, CkfBaseTrajectoryFilter, "CkfBaseTrajectoryFilter");
DEFINE_EDM_PLUGIN(TrajectoryFilterFactory, ChargeSignificanceTrajectoryFilter, "ChargeSignificanceTrajectoryFilter");
		  

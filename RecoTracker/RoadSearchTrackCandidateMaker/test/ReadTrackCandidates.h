//
// Package:         RecoTracker/RoadSearchTrackCandidateMaker/test
// Class:           ReadTrackCandidates
// 
// Description:     calls ReadTrackCandidatesAlgorithm to
//                  read and analyze TrackCandidates from the RoadSearch
//
// Original Author: Oliver Gutsche, gutsche@fnal.gov
// Created:         Wed Mar 29 20:00:00 UTC 2006
//
// $Author: gutsche $
// $Date: 2007/03/07 21:46:52 $
// $Revision: 1.2 $
//

#ifndef ReadTrackCandidates_h
#define ReadTrackCandidates_h

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"


#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoTracker/RoadSearchTrackCandidateMaker/test/ReadTrackCandidatesAlgorithm.h"

namespace cms
{
  class ReadTrackCandidates : public edm::EDAnalyzer
  {
  public:

    explicit ReadTrackCandidates(const edm::ParameterSet& conf);

    virtual ~ReadTrackCandidates();

    virtual void analyze(const edm::Event& e, const edm::EventSetup& c);

  private:
    ReadTrackCandidatesAlgorithm readTrackCandidatesAlgorithm_;
    edm::ParameterSet conf_;

  };
}


#endif

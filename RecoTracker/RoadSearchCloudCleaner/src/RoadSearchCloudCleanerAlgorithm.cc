//
// Package:         RecoTracker/RoadSearchCloudMaker
// Class:           RoadSearchCloudCleanerAlgorithm
// 
// Description:     
//
// Original Author: Steve Wagner, stevew@pizero.colorado.edu
// Created:         Sat Feb 19 22:00:00 UTC 2006
//
// $Author: gutsche $
// $Date: 2006/03/28 23:12:10 $
// $Revision: 1.1 $
//

#include <vector>

#include "RecoTracker/RoadSearchCloudCleaner/interface/RoadSearchCloudCleanerAlgorithm.h"

#include "DataFormats/RoadSearchCloud/interface/RoadSearchCloud.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

RoadSearchCloudCleanerAlgorithm::RoadSearchCloudCleanerAlgorithm(const edm::ParameterSet& conf) : conf_(conf) { 
}

RoadSearchCloudCleanerAlgorithm::~RoadSearchCloudCleanerAlgorithm() {
}

void RoadSearchCloudCleanerAlgorithm::run(const RoadSearchCloudCollection* input,
					  const edm::EventSetup& es,
					  RoadSearchCloudCollection &output)
{

  //
  //  right now cloud cleaning solely consist of merging clouds based on the number
  //  of shared hits (getting rid of obvious duplicates) - don't need roads and
  //  geometry for this - eventually this stage will become a sub-process (probably
  //  early on) of cloud cleaning - but for now that's all, folks
  //

  edm::LogInfo("RoadSearch") << "Raw Clouds input size: " << input->size(); 

  //
  //  no raw clouds - nothing to try merging
  //

  if ( input->empty() ){
  //  if ( conf_.getUntrackedParameter<int>("VerbosityLevel") > 0 ) {
      LogDebug("RoadSearch") << "Found " << output.size() << " clouds.";
  //  }
    return;  
  }

  //
  //  1 raw cloud - nothing to try merging, but one cloud to duplicate
  //

  if ( 1==input->size() ){
    output.push_back(*(input->begin()->clone()));
    LogDebug("RoadSearch") << "Found " << output.size() << " clouds.";
    return;
  }  

  //
  //  got > 1 raw cloud - something to try merging
  //
  double mergingFraction = conf_.getParameter<double>("MergingFraction");
  unsigned int maxRecHitsInCloud = conf_.getParameter<int>("MaxRecHitsInCloud");
  std::vector<bool> dont_merge;
  std::vector<bool> already_gone;
  for (unsigned int i=0; i<input->size(); ++i) {
    already_gone.push_back(false); 
    dont_merge.push_back(false);
  }

  int raw_cloud_ctr=0;
  // loop over clouds
  for ( RoadSearchCloudCollection::const_iterator raw_cloud = input->begin(); raw_cloud != input->end(); ++raw_cloud) {
    ++raw_cloud_ctr;

    if (already_gone[raw_cloud_ctr-1])continue;
    LogDebug("RoadSearch") << "number of ref in rawcloud " << raw_cloud->seeds().size(); 

    // produce output cloud where other clouds are merged in
    RoadSearchCloud lone_cloud;
    RoadSearchCloud::RecHitOwnVector raw_hits = raw_cloud->recHits();
    for (unsigned int i=0; i<raw_hits.size(); ++i) {
      lone_cloud.addHit(raw_hits[i].clone());
    }
    RoadSearchCloud::SeedRefs raw_seed_refs = raw_cloud->seeds();
    for ( RoadSearchCloud::SeedRefs::const_iterator rawseedref = raw_seed_refs.begin();
	  rawseedref != raw_seed_refs.end();
	  ++rawseedref ) {
      RoadSearchCloud::SeedRef ref = *rawseedref;
      lone_cloud.addSeed(ref);
    }

    int second_cloud_ctr=0;
    for ( RoadSearchCloudCollection::const_iterator second_cloud = input->begin(); second_cloud != input->end(); ++second_cloud) {
      second_cloud_ctr++;

      if ( already_gone[second_cloud_ctr-1] || dont_merge[raw_cloud_ctr-1] )continue;
      LogDebug("RoadSearch") << "number of ref in second_cloud " << second_cloud->seeds().size(); 

      if (second_cloud_ctr > raw_cloud_ctr){

        RoadSearchCloud::RecHitOwnVector lone_cloud_hits = lone_cloud.recHits();
        RoadSearchCloud::RecHitOwnVector second_cloud_hits = second_cloud->recHits();
        RoadSearchCloud::RecHitOwnVector unsharedhits;
        for (unsigned int j=0; j<second_cloud_hits.size(); ++j) {
          bool is_shared=false;
          for (unsigned int i=0; i<lone_cloud_hits.size(); ++i) {

	    // compare two TrackingRecHits, if they would be refs, you could compare the refs,
	    // temp. solution, compare rechits by detid and localposition
	    // change later

            if (lone_cloud_hits[i].geographicalId() == second_cloud_hits[j].geographicalId())
	      if (lone_cloud_hits[i].localPosition().x() == second_cloud_hits[j].localPosition().x())
		if (lone_cloud_hits[i].localPosition().y() == second_cloud_hits[j].localPosition().y())
		  if (lone_cloud_hits[i].localPosition().z() == second_cloud_hits[j].localPosition().z())
		    {is_shared=true; break;}
          }
          if (!is_shared)unsharedhits.push_back(second_cloud_hits[j].clone());
        }

        float f_lone_shared=float(second_cloud->size()-unsharedhits.size())/float(lone_cloud.size());
        float f_second_shared=float(second_cloud->size()-unsharedhits.size())/float(second_cloud->size());

        if ( ( (f_lone_shared > mergingFraction)||(f_second_shared > mergingFraction) ) 
	     && (lone_cloud.size()+unsharedhits.size() <= maxRecHitsInCloud) ){

	  LogDebug("RoadSearch") << "Add clouds.";
	  
	  //
	  //  got a cloud to merge
	  //
          for (unsigned int k=0; k<unsharedhits.size(); ++k) {
            lone_cloud.addHit(unsharedhits[k].clone());
          }

	  // add seed of second_cloud to lone_cloud
	  RoadSearchCloud::SeedRefs second_seed_refs = second_cloud->seeds();
	  for ( RoadSearchCloud::SeedRefs::const_iterator secondseedref = second_seed_refs.begin();
		secondseedref != second_seed_refs.end();
		++secondseedref ) {
	    RoadSearchCloud::SeedRef ref = *secondseedref;
	    lone_cloud.addSeed(ref);
	  }

          already_gone[second_cloud_ctr-1]=true;

        }//end got a cloud to merge

      }//second cloud acceptable for inspection

    }//interate over all second clouds

    LogDebug("RoadSearch") << "number of ref in cloud " << lone_cloud.seeds().size(); 

    output.push_back(lone_cloud);

  }//iterate over all raw clouds

  edm::LogInfo("RoadSearch") << "Found " << output.size() << " clouds.";

};



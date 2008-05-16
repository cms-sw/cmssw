#ifndef RecoTracker_SpecialSeedGenerators_ClusterChecker_H
#define RecoTracker_SpecialSeedGenerators_ClusterChecker_H

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/LazyGetter.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/Common/interface/DetSetVector.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

class ClusterChecker {
 public: 
  ClusterChecker(edm::ParameterSet conf) :
    doACheck_(conf.getParameter<bool>("doClusterCheck"))
    {
      if (doACheck_){
	clusterCollectionInputTag_ = conf.getParameter<edm::InputTag>("ClusterCollectionLabel");
	maxNrOfCosmicClusters_ = conf.getParameter<unsigned int>("MaxNumberOfCosmicClusters");
      }
    }
    
  bool tooManyClusters(const edm::Event & e){
    if (!doACheck_) return false;

    // get special input for cosmic cluster multiplicity filter
    edm::Handle<edm::DetSetVector<SiStripCluster> > clusterDSV;
    e.getByLabel(clusterCollectionInputTag_, clusterDSV);
    bool tooManyClusters = false;
    if (!clusterDSV.failedToGet()) {
      const edm::DetSetVector<SiStripCluster> & input = *clusterDSV;

      unsigned int totalClusters = 0;
      //loop over detectors
      edm::DetSetVector<SiStripCluster>::const_iterator DSViter=input.begin(), DSViter_end=input.end();
      for (; DSViter!=DSViter_end; DSViter++ ) {
	totalClusters+=DSViter->data.size();
	if (totalClusters>maxNrOfCosmicClusters_) break;
      }
      tooManyClusters = (totalClusters>maxNrOfCosmicClusters_);
    }
    else{
      edm::Handle<edm::LazyGetter<SiStripCluster> > lazyGH;
      e.getByLabel(clusterCollectionInputTag_, lazyGH);
      if (!lazyGH.failedToGet()){
	tooManyClusters = (lazyGH->size()>maxNrOfCosmicClusters_);
      }else{
	//say something's wrong.
	edm::LogError("ClusterChecker")<<"could not get any SiStrip cluster collections of type edm::DetSetVector<SiStripCluster> or edm::LazyGetter<SiStripCluster, with label: "<<clusterCollectionInputTag_;
	tooManyClusters = true;
      }
    }
    return tooManyClusters;
  }

 private: 
  bool doACheck_;
  edm::InputTag clusterCollectionInputTag_;
  uint maxNrOfCosmicClusters_;
};

#endif

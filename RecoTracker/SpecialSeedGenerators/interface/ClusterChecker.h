#ifndef RecoTracker_SpecialSeedGenerators_ClusterChecker_H
#define RecoTracker_SpecialSeedGenerators_ClusterChecker_H

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/LazyGetter.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

class ClusterChecker {
 public: 
  ClusterChecker(edm::ParameterSet conf) :
    doACheck_(conf.getParameter<bool>("doClusterCheck"))
    {
      if (doACheck_){
	clusterCollectionInputTag_ = conf.getParameter<edm::InputTag>("ClusterCollectionLabel");
	maxNrOfCosmicClusters_ = conf.getParameter<unsigned int>("MaxNumberOfCosmicClusters");
        if (conf.existsAs<uint32_t>("DontCountDetsAboveNClusters")) {
            ignoreDetsAboveNClusters_ = conf.getParameter<uint32_t>("DontCountDetsAboveNClusters");
        } else {
            ignoreDetsAboveNClusters_ = 0;
        }
      }
    }
    
  size_t tooManyClusters(const edm::Event & e){
    if (!doACheck_) return 0;

    // get special input for cosmic cluster multiplicity filter
    edm::Handle<edmNew::DetSetVector<SiStripCluster> > clusterDSV;
    e.getByLabel(clusterCollectionInputTag_, clusterDSV);
    unsigned int totalClusters = 0;
    if (!clusterDSV.failedToGet()) {
      const edmNew::DetSetVector<SiStripCluster> & input = *clusterDSV;

      if (ignoreDetsAboveNClusters_ == 0) {
        totalClusters = input.dataSize();
      } else {
          //loop over detectors
          edmNew::DetSetVector<SiStripCluster>::const_iterator DSViter=input.begin(), DSViter_end=input.end();
          for (; DSViter!=DSViter_end; DSViter++ ) {
            size_t siz = DSViter->size();
            if (siz > ignoreDetsAboveNClusters_) continue;
            totalClusters += siz; 
          }
      }
    }
    else{
      edm::Handle<edm::LazyGetter<SiStripCluster> > lazyGH;
      e.getByLabel(clusterCollectionInputTag_, lazyGH);
      if (!lazyGH.failedToGet()){
        totalClusters = lazyGH->size();
      }else{
	//say something's wrong.
	edm::LogError("ClusterChecker")<<"could not get any SiStrip cluster collections of type edm::DetSetVector<SiStripCluster> or edm::LazyGetter<SiStripCluster, with label: "<<clusterCollectionInputTag_;
        totalClusters = 999999;
      }
    }
    return (totalClusters > maxNrOfCosmicClusters_) ? totalClusters : 0;
  }

 private: 
  bool doACheck_;
  edm::InputTag clusterCollectionInputTag_;
  uint32_t maxNrOfCosmicClusters_;
  uint32_t ignoreDetsAboveNClusters_;
};

#endif

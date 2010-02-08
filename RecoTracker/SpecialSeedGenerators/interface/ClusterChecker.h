#ifndef RecoTracker_SpecialSeedGenerators_ClusterChecker_H
#define RecoTracker_SpecialSeedGenerators_ClusterChecker_H

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/LazyGetter.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
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
	pixelClusterCollectionInputTag_ = conf.getParameter<edm::InputTag>("PixelClusterCollectionLabel");
	maxNrOfPixelClusters_ = conf.getParameter<unsigned int>("MaxNumberOfPixelClusters");
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
    if (totalClusters > maxNrOfCosmicClusters_) return totalClusters;

    // get special input for pixel cluster multiplicity filter
    edm::Handle<edmNew::DetSetVector<SiPixelCluster> > pixelClusterDSV;
    e.getByLabel(pixelClusterCollectionInputTag_, pixelClusterDSV);
    unsigned int totalPixelClusters = 0;
    if (!pixelClusterDSV.failedToGet()) {
      const edmNew::DetSetVector<SiPixelCluster> & input = *pixelClusterDSV;

      if (ignoreDetsAboveNClusters_ == 0) {
        totalPixelClusters = input.dataSize();
      } else {
          //loop over detectors
          edmNew::DetSetVector<SiPixelCluster>::const_iterator DSViter=input.begin(), DSViter_end=input.end();
          for (; DSViter!=DSViter_end; DSViter++ ) {
            size_t siz = DSViter->size();
            if (siz > ignoreDetsAboveNClusters_) continue;
            totalPixelClusters += siz; 
          }
      }
    }
    else{
      //say something's wrong.
      edm::LogError("ClusterChecker")<<"could not get any SiPixel cluster collections of type edm::DetSetVector<SiPixelCluster>  with label: "<<pixelClusterCollectionInputTag_;
      totalPixelClusters = 999999;
    }
    if (totalPixelClusters > maxNrOfPixelClusters_) return totalPixelClusters;

    return 0;

  }

 private: 
  bool doACheck_;
  edm::InputTag clusterCollectionInputTag_;
  edm::InputTag pixelClusterCollectionInputTag_;
  uint32_t maxNrOfCosmicClusters_;
  uint32_t maxNrOfPixelClusters_;
  uint32_t ignoreDetsAboveNClusters_;
};

#endif

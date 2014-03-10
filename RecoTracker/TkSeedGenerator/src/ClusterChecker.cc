#include "RecoTracker/TkSeedGenerator/interface/ClusterChecker.h"

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

ClusterChecker::ClusterChecker(const edm::ParameterSet & conf,
	edm::ConsumesCollector && iC):
  ClusterChecker(conf, iC)
{}

ClusterChecker::ClusterChecker(const edm::ParameterSet & conf,
	edm::ConsumesCollector & iC):
    doACheck_(conf.getParameter<bool>("doClusterCheck")),
    selector_(conf.getParameter<bool>("doClusterCheck") && conf.existsAs<std::string>("cut") ?
                conf.getParameter<std::string>("cut") : 
                "")
{
    if (doACheck_){
        clusterCollectionInputTag_ = conf.getParameter<edm::InputTag>("ClusterCollectionLabel");
        pixelClusterCollectionInputTag_ = conf.getParameter<edm::InputTag>("PixelClusterCollectionLabel");
	token_sc = iC.consumes<edmNew::DetSetVector<SiStripCluster> >(clusterCollectionInputTag_);
	token_pc = iC.consumes<edmNew::DetSetVector<SiPixelCluster> >(pixelClusterCollectionInputTag_);
        maxNrOfCosmicClusters_     = conf.getParameter<unsigned int>("MaxNumberOfCosmicClusters");
        maxNrOfPixelClusters_ = conf.getParameter<unsigned int>("MaxNumberOfPixelClusters");
        if (conf.existsAs<uint32_t>("DontCountDetsAboveNClusters")) {
            ignoreDetsAboveNClusters_ = conf.getParameter<uint32_t>("DontCountDetsAboveNClusters");
        } else {
            ignoreDetsAboveNClusters_ = 0;
        }
    }
}


ClusterChecker::~ClusterChecker()
{
}

size_t ClusterChecker::tooManyClusters(const edm::Event & e) const  
{
    if (!doACheck_) return 0;

    // get special input for cosmic cluster multiplicity filter
    edm::Handle<edmNew::DetSetVector<SiStripCluster> > clusterDSV;
    e.getByToken(token_sc, clusterDSV);
    reco::utils::ClusterTotals totals;
    if (!clusterDSV.failedToGet()) {
        const edmNew::DetSetVector<SiStripCluster> & input = *clusterDSV;

        if (ignoreDetsAboveNClusters_ == 0) {
            totals.strip = input.dataSize();
            totals.stripdets = input.size();
        } else {
            //loop over detectors
            totals.strip = 0;
            totals.stripdets = 0;
            edmNew::DetSetVector<SiStripCluster>::const_iterator DSViter=input.begin(), DSViter_end=input.end();
            for (; DSViter!=DSViter_end; DSViter++ ) {
                size_t siz = DSViter->size();
                if (siz > ignoreDetsAboveNClusters_) continue;
                totals.strip += siz; 
                totals.stripdets++;
            }
        }
    }
    else{
        edm::Handle<edm::LazyGetter<SiStripCluster> > lazyGH;
        e.getByToken(token_sc, lazyGH);
        totals.stripdets = 0; // don't know how to count this online
        if (!lazyGH.failedToGet()){
            totals.strip = lazyGH->size();
        }else{
            //say something's wrong.
            edm::LogError("ClusterChecker")<<"could not get any SiStrip cluster collections of type edm::DetSetVector<SiStripCluster> or edm::LazyGetter<SiStripCluster, with label: "<<clusterCollectionInputTag_;
            totals.strip = 999999;
        }
    }
    if (totals.strip > int(maxNrOfCosmicClusters_)) return totals.strip;

    // get special input for pixel cluster multiplicity filter
    edm::Handle<edmNew::DetSetVector<SiPixelCluster> > pixelClusterDSV;
    e.getByToken(token_pc, pixelClusterDSV);
    if (!pixelClusterDSV.failedToGet()) {
        const edmNew::DetSetVector<SiPixelCluster> & input = *pixelClusterDSV;

        if (ignoreDetsAboveNClusters_ == 0) {
            totals.pixel = input.dataSize();
            totals.pixeldets = input.size();
        } else {
            //loop over detectors
            totals.pixel = 0;
            totals.pixeldets = 0;
            edmNew::DetSetVector<SiPixelCluster>::const_iterator DSViter=input.begin(), DSViter_end=input.end();
            for (; DSViter!=DSViter_end; DSViter++ ) {
                size_t siz = DSViter->size();
                if (siz > ignoreDetsAboveNClusters_) continue;
                totals.pixel += siz; 
                totals.pixeldets++;
            }
        }
    }
    else{
        //say something's wrong.
        edm::LogError("ClusterChecker")<<"could not get any SiPixel cluster collections of type edm::DetSetVector<SiPixelCluster>  with label: "<<pixelClusterCollectionInputTag_;
        totals.pixel = 999999;
    }
    if (totals.pixel > int(maxNrOfPixelClusters_)) return totals.pixel;

    if (!selector_(totals)) return totals.strip;
    return 0;
}

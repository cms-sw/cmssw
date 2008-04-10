#include "RecoTracker/TrackProducer/interface/ClusterRemovalRefSetter.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"

ClusterRemovalRefSetter::ClusterRemovalRefSetter(const edm::Event &iEvent, const edm::InputTag tag) {
    edm::Handle<reco::ClusterRemovalInfo> hCRI;
    iEvent.getByLabel(tag, hCRI);
    cri_ = &*hCRI; 

    iEvent.get(cri_->pixelProdID(), handlePixel_);
    iEvent.get(cri_->stripProdID(), handleStrip_);

    //std::cout << "Rekeying PIXEL ProdID " << cri_->pixelNewProdID() << " => " << cri_->pixelProdID() << std::endl;
    //std::cout << "Rekeying STRIP ProdID " << cri_->stripNewProdID() << " => " << cri_->stripProdID() << std::endl;
}

void ClusterRemovalRefSetter::reKey(TrackingRecHit *hit) const {
    if (!hit->isValid()) return;
    DetId detid = hit->geographicalId(); 
    uint32_t subdet = detid.subdetId();
    if ((subdet == PixelSubdetector::PixelBarrel) || (subdet == PixelSubdetector::PixelEndcap)) {
        reKey(reinterpret_cast<SiPixelRecHit *>(hit), detid.rawId());
    } else {
        const std::type_info &type = typeid(*hit);
        if (type == typeid(SiStripRecHit2D)) {
            reKey(reinterpret_cast<SiStripRecHit2D *>(hit), detid.rawId());
        } else if (type == typeid(SiStripMatchedRecHit2D)) {
            SiStripMatchedRecHit2D *mhit = reinterpret_cast<SiStripMatchedRecHit2D *>(hit);
            // const_cast is needed: monoHit() e stereoHit() are const only - at least for now
            reKey(mhit->monoHit(), mhit->monoHit()->geographicalId().rawId());
            reKey(mhit->stereoHit(), mhit->stereoHit()->geographicalId().rawId());
        } else if (type == typeid(ProjectedSiStripRecHit2D)) {
            ProjectedSiStripRecHit2D *phit = reinterpret_cast<ProjectedSiStripRecHit2D *>(hit);
            reKey(&phit->originalHit(), phit->originalHit().geographicalId().rawId());
        } else throw cms::Exception("Unknown RecHit Type") << "RecHit of type " << type.name() << " not supported. (use c++filt to demangle the name)";
    }
}

void ClusterRemovalRefSetter::reKey(SiStripRecHit2D *hit, uint32_t detid) const {
    using reco::ClusterRemovalInfo;
    const ClusterRemovalInfo::Indices &indices = cri_->stripIndices();
    SiStripRecHit2D::ClusterRef newRef = hit->cluster();
    // "newRef" as it refs to the "new"(=cleaned) collection, instead of the old one

    if (newRef.id() != cri_->stripNewProdID()) {   // this is a cfg error in the tracking configuration, much more likely
        throw cms::Exception("Inconsistent Data") << "ClusterRemovalRefSetter: " << 
            "Existing strip cluster refers to product ID " << newRef.id() << 
            " while the ClusterRemovalInfo expects as *new* cluster collection the ID " << cri_->stripNewProdID() << "\n";
    }

    size_t newIndex = newRef.key();
    assert(newIndex < indices.size());
    size_t oldIndex = indices[newIndex];
    SiStripRecHit2D::ClusterRef oldRef(handleStrip_, oldIndex, false);
    hit->setClusterRef(oldRef);
}

void ClusterRemovalRefSetter::reKey(SiPixelRecHit *hit, uint32_t detid) const {
    using reco::ClusterRemovalInfo;
    const ClusterRemovalInfo::Indices &indices = cri_->pixelIndices();
    SiPixelRecHit::ClusterRef newRef = hit->cluster();
    // "newRef" as it refs to the "new"(=cleaned) collection, instead of the old one

    if (newRef.id()  != cri_->pixelNewProdID()) {
        throw cms::Exception("Inconsistent Data") << "ClusterRemovalRefSetter: " << 
            "Existing pixel cluster refers to product ID " << newRef.id() << 
            " while the ClusterRemovalInfo expects as *new* cluster collection the ID " << cri_->pixelNewProdID() << "\n";
    }
    size_t newIndex = newRef.key();
    assert(newIndex < indices.size());
    size_t oldIndex = indices[newIndex];
    SiPixelRecHit::ClusterRef oldRef(handlePixel_, oldIndex, false);
    hit->setClusterRef(oldRef);
}



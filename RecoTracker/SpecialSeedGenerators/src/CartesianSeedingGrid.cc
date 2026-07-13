#include "RecoTracker/SpecialSeedGenerators/interface/CartesianSeedingGrid.h"


CartesianSeedingGrid::CartesianSeedingGrid (int nBinsX, double xmin, double xmax, int nBinsY, double ymin, double ymax,int nBinsZ, double zmin, double zmax ):
    nBinsX_(nBinsX),
    xmin_(xmin), 
    xmax_(xmax), 
    nBinsY_(nBinsY),
    ymin_(ymin), 
    ymax_(ymax), 
    nBinsZ_(nBinsZ),
    zmin_(zmin), 
    zmax_(zmax){
    recHits_.resize(nBinsX*nBinsY*nBinsZ);
    }

void CartesianSeedingGrid::addHit(const BaseTrackerRecHit* h){
    recHits_.at(getBin(binX(h->globalPosition().x()), binY(h->globalPosition().y()), binZ(h->globalPosition().z()))).push_back(h);
}

const std::vector<const BaseTrackerRecHit*> & CartesianSeedingGrid::getHits(int binX, int binY, int binZ) const{
    return recHits_.at(getBin(binX,binY,binZ));
}

void CartesianSeedingGrid::sort(){
    for (auto & cell : recHits_){
    std::sort(cell.begin(),cell.end(),[](const BaseTrackerRecHit* h1, const BaseTrackerRecHit* h2){return h1->globalPosition().y() < h2->globalPosition().y(); });
    }
}

int CartesianSeedingGrid::getBin(int bx, int by, int bz) const{
    return bz + nBinsZ_ * (by + nBinsY_ * bx);
}

int CartesianSeedingGrid::findBin(double value, int nBins, double min, double max) const {
    return std::clamp<int>(std::floor(( value- min) / (max - min) * nBins),0,nBins-1);
}


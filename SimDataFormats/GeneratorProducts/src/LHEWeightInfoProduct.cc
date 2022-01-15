#include <vector>
#include <string>

#include "SimDataFormats/GeneratorProducts/interface/LHEWeightInfoProduct.h"

LHEWeightInfoProduct::LHEWeightInfoProduct(edm::OwnVector<gen::WeightGroupInfo>& weightGroups) {
    weightGroupsInfo_ = weightGroups;
}
        
LHEWeightInfoProduct& LHEWeightInfoProduct::operator=(const LHEWeightInfoProduct &other) {
    weightGroupsInfo_ = other.weightGroupsInfo_; 
    return * this; 
}

LHEWeightInfoProduct& LHEWeightInfoProduct::operator=(LHEWeightInfoProduct &&other) {
    weightGroupsInfo_ = std::move(other.weightGroupsInfo_); 
    return *this;
}

const edm::OwnVector<gen::WeightGroupInfo>& LHEWeightInfoProduct::allWeightGroupsInfo() const { 
    return weightGroupsInfo_; 
}

const gen::WeightGroupInfo& LHEWeightInfoProduct::containingWeightGroupInfo(int index) const {
    for (const auto& weightGroup : weightGroupsInfo_) {
        if (weightGroup.indexInRange(index))
            return weightGroup;
    }
    throw std::domain_error("Failed to find containing weight group");
}

const gen::WeightGroupInfo& LHEWeightInfoProduct::orderedWeightGroupInfo(int weightGroupIndex) const {
    if (weightGroupIndex >= static_cast<int>(weightGroupsInfo_.size()))
        throw std::range_error("Weight index out of range!");
    return weightGroupsInfo_[weightGroupIndex];
}

void LHEWeightInfoProduct::addWeightGroupInfo(gen::WeightGroupInfo& info) {  
    weightGroupsInfo_.push_back(info); 
}

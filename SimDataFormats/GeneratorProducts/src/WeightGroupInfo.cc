#include <string>
#include <algorithm>
#include "SimDataFormats/GeneratorProducts/interface/WeightGroupInfo.h"

namespace gen {
    void WeightGroupInfo::copy(const WeightGroupInfo &other) {
        headerEntry_ = other.headerEntry();
        name_ = other.name();
        weightType_ = other.weightType();
        idsContained_ = other.idsContained();
        firstId_ = other.firstId();
        lastId_ = other.lastId();
    }

    WeightGroupInfo* WeightGroupInfo::clone() const {
        return new WeightGroupInfo(*this);
    }

    WeightMetaInfo WeightGroupInfo::weightMetaInfo(int weightEntry) {
        return idsContained_.at(weightEntry);
    }

    WeightMetaInfo WeightGroupInfo::weightMetaInfo(std::string wgtId) {
        int weightEntry = weightVectorEntry(wgtId);
        return idsContained_.at(weightEntry);
    }

    int WeightGroupInfo::weightVectorEntry(const std::string& wgtId) {
        return weightVectorEntry(wgtId, 0);
    }

    int WeightGroupInfo::containsWeight(const std::string& wgtId, int weightEntry) {
        return weightVectorEntry(wgtId, weightEntry) != -1;
    }

    int WeightGroupInfo::weightVectorEntry(const std::string& wgtId, int weightEntry) {
        int entry = -1;
        if (!indexInRange(weightEntry)) {
            size_t orderedEntry = weightEntry - firstId_;
            if (orderedEntry < idsContained_.size())
                if (idsContained_.at(orderedEntry).id == wgtId)
                    return orderedEntry;
        }
        auto it = std::find_if(idsContained_.begin(), idsContained_.end(), 
                        [wgtId] (const WeightMetaInfo& w) { return w.id == wgtId; });
        if (it != idsContained_.end())
            return std::distance(idsContained_.begin(), it);
        return entry;
    }

    void WeightGroupInfo::addContainedId(int weightEntry, std::string id, std::string label="") {
        if (firstId_ == -1 || weightEntry < firstId_) {
            firstId_ = weightEntry;
            // Reset to reflect that indices will be shifted
            for (auto& id : idsContained_)
                id.localIndex = id.globalIndex - firstId_;
        }
        if (weightEntry > lastId_)
            lastId_ = weightEntry;
        
        WeightMetaInfo info;
        info.globalIndex = weightEntry;
        info.localIndex = weightEntry - firstId_;
        info.id = id;
        info.label = label;

        if (idsContained_.size() < info.localIndex) {
            idsContained_.resize(info.localIndex);
            idsContained_.insert(idsContained_.begin()+info.localIndex, info);
        }
        else if (idsContained_.size() == info.localIndex) {
            idsContained_.push_back(info);
        }
        else {
            idsContained_.resize(info.localIndex+1);
            idsContained_[info.localIndex] = info;
        }
    }

    std::vector<WeightMetaInfo> WeightGroupInfo::containedIds() const { return idsContained_; }


    bool WeightGroupInfo::indexInRange(int index) const {
        return (index <= lastId_ && index >= firstId_);
    }
}

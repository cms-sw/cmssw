#include "SimDataFormats/GeneratorProducts/interface/ScaleWeightGroupInfo.h"
#include <stdexcept>

namespace gen {
    void ScaleWeightGroupInfo::copy(const ScaleWeightGroupInfo &other) {
        icentral_ = centralIndex();
        imuR1muF2_ = muR1muF2Index();
        imuR1muF05_ = muR1muF05Index();
        imuR2muF05_ = muR2muF05Index();
        imuR2muF1_ = muR2muF1Index();
        imuR2muF2_ = muR2muF2Index();
        imuR2muF05_ = muR2muF05Index();
        imuR05muF1_ = muR05muF1Index();
        imuR05muF2_ = muR05muF2Index();
        WeightGroupInfo::copy(other);
    }

    ScaleWeightGroupInfo* ScaleWeightGroupInfo::clone() const {
        return new ScaleWeightGroupInfo(*this);
    }

    void ScaleWeightGroupInfo::addContainedId(int weightEntry, std::string id, std::string label, float muR, float muF) {
        WeightGroupInfo::addContainedId(weightEntry, id, label);
        auto metaInfo = weightMetaInfo(weightEntry);
        setMuRMuFIndex(metaInfo, muR, muF);
    }

    void ScaleWeightGroupInfo::setMuRMuFIndex(WeightMetaInfo info, float muR, float muF) {
        if (muR == 0.5 && muF == 0.5)
            imuR05muF05_ = info.localIndex;
        else if (muR == 0.5 && muF == 1.0)
            imuR05muF1_ = info.localIndex;
        else if (muR == 0.5 && muF == 2.0)
            imuR05muF2_ = info.localIndex;
        else if (muR == 1.0 && muF == 0.5)
            imuR1muF05_ = info.localIndex;
        else if (muR == 1.0 && muF == 1.0)
            icentral_ = info.localIndex;
        else if (muR == 1.0 && muF == 2.0)
            imuR1muF2_ = info.localIndex;
        else if (muR == 2.0 && muF == 0.5)
            imuR2muF05_ = info.localIndex;
        else if (muR == 2.0 && muF == 1.0)
            imuR2muF1_ = info.localIndex;
        else if (muR == 2.0 && muF == 2.0)
            imuR2muF2_ = info.localIndex;
        else
            throw std::invalid_argument("Invalid muF and muR variation is not a factor of two from central value");
    }
}



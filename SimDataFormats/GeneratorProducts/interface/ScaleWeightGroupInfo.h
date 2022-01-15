#ifndef SimDataFormats_GeneratorProducts_ScaleWeightGroupInfo_h
#define SimDataFormats_GeneratorProducts_ScaleWeightGroupInfo_h

#include "SimDataFormats/GeneratorProducts/interface/WeightGroupInfo.h"

namespace gen {
    class ScaleWeightGroupInfo : public WeightGroupInfo {
        private:
            bool isFuncationFormVar_;
            size_t icentral;
            size_t imuR1muF2;
            size_t imuR1muF05;
            size_t imuR2muF05;
            size_t imuR2muF1;
            size_t imuR2muF2;
            size_t imuR05muF05;
            size_t imuR05muF1;
            size_t imuR05muF2;
        public:
            ScaleWeightGroupInfo() : ScaleWeightGroupInfo("") {}
	        ScaleWeightGroupInfo(std::string header, std::string name) : 
                WeightGroupInfo(header, name) { 
                weightType_ = kScaleWeights;
                isFuncationFormVar_ = false;
                icentral = 0;
                imuR1muF2 = 0;
                imuR1muF05 = 0;
                imuR2muF05 = 0;
                imuR2muF1 = 0;
                imuR2muF2 = 0;
                imuR2muF05 = 0;
                imuR05muF05 = 0;
                imuR05muF1 = 0;
                imuR05muF2 = 0;
            }
	        ScaleWeightGroupInfo(std::string header) : 
                ScaleWeightGroupInfo(header, header) { } 
            ScaleWeightGroupInfo(const ScaleWeightGroupInfo &other) {
                copy(other);
            }
            virtual ~ScaleWeightGroupInfo() override {}
            void copy(const ScaleWeightGroupInfo &other);
            ScaleWeightGroupInfo* clone() const;

            void setMuRMuFIndex(WeightMetaInfo info, float muR, float muF);
            void addContainedId(int weightEntry, std::string id, std::string label, float muR, float muF);

            // Is a variation of the functional form of the dynamic scale
            bool isFunctionalFormVariation();
            void setIsFunctionalFormVariation(bool functionalVar) {isFuncationFormVar_ = functionalVar; }
            size_t centralIndex() {return icentral; }
            size_t muR1muF2Index() { return imuR1muF2; }
            size_t muR1muF05Index() { return imuR1muF05; }
            size_t muR2muF05Index() { return imuR2muF05; }
            size_t muR2muF1Index() { return imuR2muF1; }
            size_t muR2muF2Index() { return imuR2muF2; }
            size_t muR05muF05Index() { return imuR05muF05; }
            size_t muR05muF1Index() { return imuR05muF1; }
            size_t muR05muF2Index() { return imuR05muF2; }
    };
}

#endif


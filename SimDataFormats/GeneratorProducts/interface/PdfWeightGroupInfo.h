#ifndef SimDataFormats_GeneratorProducts_PdfWeightGroupInfo_h
#define SimDataFormats_GeneratorProducts_PdfWeightGroupInfo_h

#include "SimDataFormats/GeneratorProducts/interface/WeightGroupInfo.h"

namespace gen {
    enum PdfUncertaintyType {
        kHessianUnc,
        kMonteCarloUnc,
        kUnknownUnc,
    };

    class PdfWeightGroupInfo : public WeightGroupInfo {
        private:
            PdfUncertaintyType uncertaintyType_;
            bool hasAlphasVars_;
            int alphasUpIndex_;
            int alphasDownIndex_;
        public:
            PdfWeightGroupInfo() : WeightGroupInfo() { weightType_ = kPdfWeights; }
	        PdfWeightGroupInfo(std::string header, std::string name) : 
                WeightGroupInfo(header, name) { weightType_ = kPdfWeights; } 
	        PdfWeightGroupInfo(std::string header) : 
                WeightGroupInfo(header) { weightType_ = kPdfWeights; } 
            PdfWeightGroupInfo(const PdfWeightGroupInfo &other) {
                copy(other);
            }
            virtual ~PdfWeightGroupInfo() override {}
            void copy(const PdfWeightGroupInfo &other);
            PdfWeightGroupInfo* clone() const;

            void setUncertaintyType(PdfUncertaintyType uncertaintyType) { uncertaintyType_ = uncertaintyType; }
            void setHasAlphasVariations(bool hasAlphasVars) { hasAlphasVars_ = hasAlphasVars; }
            void setAlphasUpIndex(int alphasUpIndex) { alphasUpIndex_ = alphasUpIndex; }
            void setAlphasDownIndex(int alphasDownIndex) { alphasDownIndex_ = alphasDownIndex; }
            PdfUncertaintyType uncertaintyType() const { return uncertaintyType_; }
            bool hasAlphasVariations() const { return hasAlphasVars_; }
            int alphasUpIndex() const { return alphasUpIndex_; }
            int alphasDownIndex() const { return alphasDownIndex_; }
    };
}

#endif // SimDataFormats_GeneratorProducts_PdfWeightGroupInfo_h


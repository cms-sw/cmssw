#include "SimDataFormats/GeneratorProducts/interface/PdfWeightGroupInfo.h"

namespace gen {
    void PdfWeightGroupInfo::copy(const PdfWeightGroupInfo &other) {
        uncertaintyType_ = other.uncertaintyType();
        hasAlphasVars_ = other.hasAlphasVariations();
        alphasUpIndex_ = other.alphasDownIndex();
        alphasDownIndex_ = other.alphasDownIndex();
        WeightGroupInfo::copy(other);
    }

    PdfWeightGroupInfo* PdfWeightGroupInfo::clone() const {
        return new PdfWeightGroupInfo(*this);
    }
}

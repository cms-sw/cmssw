#ifndef SimDataFormats_GeneratorProducts_PdfWeightGroupInfo_h
#define SimDataFormats_GeneratorProducts_PdfWeightGroupInfo_h

#include "SimDataFormats/GeneratorProducts/interface/WeightGroupInfo.h"
#include "LHAPDF/LHAPDF.h"
#include <iostream>
#include <set>
#include <unordered_map>

namespace gen {
  enum PdfUncertaintyType {
    kHessianUnc,
    kMonteCarloUnc,
    kVariationSet,
    kUnknownUnc,
  };

  class PdfWeightGroupInfo : public WeightGroupInfo {
  private:
    PdfUncertaintyType uncertaintyType_;
    bool hasAlphasVars_;
    int alphasUpIndex_;
    int alphasDownIndex_;
    int parentLhapdfId_ = -1;
    size_t parentLhapdfSize_ = -1;
    std::string parentLhapdfError_;
    std::vector<int> lhaids_;
    int parentLhapdfId(int lhaid) const { return lhaid - LHAPDF::lookupPDF(lhaid).second; }

  public:
    PdfWeightGroupInfo() : WeightGroupInfo() { weightType_ = WeightType::kPdfWeights; }
    PdfWeightGroupInfo(std::string header, std::string name) : WeightGroupInfo(header, name) {
      weightType_ = WeightType::kPdfWeights;
    }
    PdfWeightGroupInfo(std::string header) : WeightGroupInfo(header) { weightType_ = WeightType::kPdfWeights; }
    PdfWeightGroupInfo(const PdfWeightGroupInfo& other) : WeightGroupInfo(other) { copy(other); }
    ~PdfWeightGroupInfo() override {}
    void copy(const PdfWeightGroupInfo& other);
    PdfWeightGroupInfo* clone() const override;

    void setUncertaintyType(PdfUncertaintyType uncertaintyType) { uncertaintyType_ = uncertaintyType; }
    void setHasAlphasVariations(bool hasAlphasVars) { hasAlphasVars_ = hasAlphasVars; }
    void setAlphasUpIndex(int alphasUpIndex) { alphasUpIndex_ = alphasUpIndex; }
    void setAlphasDownIndex(int alphasDownIndex) { alphasDownIndex_ = alphasDownIndex; }
    PdfUncertaintyType uncertaintyType() const { return uncertaintyType_; }
    bool hasAlphasVariations() const { return hasAlphasVars_; }
    void addLhaid(int lhaid);
    std::vector<int>& lhaIds() { return lhaids_; }

    bool isIdInParentSet(int lhaid) const { return parentLhapdfId_ == parentLhapdfId(lhaid); }
    int parentLhapdfId() const { return parentLhapdfId_; }
    void setParentLhapdfInfo(int lhaid);

    // need to remove
    bool containsLhapdfId(int lhaid) const { return isIdInParentSet(lhaid); }

    int alphasUpIndex() const { return alphasUpIndex_; }
    int alphasDownIndex() const { return alphasDownIndex_; }
  };
}  // namespace gen

#endif  // SimDataFormats_GeneratorProducts_PdfWeightGroupInfo_h

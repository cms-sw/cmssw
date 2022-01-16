#ifndef SimDataFormats_GeneratorProducts_WeightGroupInfo_h
#define SimDataFormats_GeneratorProducts_WeightGroupInfo_h

/** \class PdfInfo
 *
 */
#include <string>
#include <algorithm>
#include <vector>
#include <array>
#include <unordered_map>

namespace gen {
  struct WeightMetaInfo {
    size_t globalIndex;
    size_t localIndex;
    std::string id;
    std::string label;

    bool operator==(const WeightMetaInfo& other) {
      return (other.globalIndex == globalIndex && other.localIndex == localIndex && other.id == id &&
              other.label == label);
    }
  };

  enum class WeightType : char {
    kPdfWeights = 'P',
    kScaleWeights = 's',
    kMEParamWeights = 'm',
    kPartonShowerWeights = 'p',
    kUnknownWeights = 'u',
  };

  const std::array<WeightType, 5> allWeightTypes = {{
      WeightType::kPdfWeights,
      WeightType::kScaleWeights,
      WeightType::kMEParamWeights,
      WeightType::kPartonShowerWeights,
      WeightType::kUnknownWeights,
  }};

  class WeightGroupInfo {
  public:
    WeightGroupInfo() : WeightGroupInfo("") {}
    WeightGroupInfo(std::string header, std::string name)
        : isWellFormed_(false), headerEntry_(header), name_(name), firstId_(-1), lastId_(-1) {}
    WeightGroupInfo(std::string header)
        : isWellFormed_(false), headerEntry_(header), name_(header), firstId_(-1), lastId_(-1) {}
    WeightGroupInfo(const WeightGroupInfo& other) { copy(other); }
    WeightGroupInfo& operator=(const WeightGroupInfo& other) {
      copy(other);
      return *this;
    }
    virtual ~WeightGroupInfo(){};
    void copy(const WeightGroupInfo& other);
    virtual WeightGroupInfo* clone() const;
    const WeightMetaInfo& weightMetaInfo(int weightEntry) const;
    const WeightMetaInfo& weightMetaInfoByGlobalIndex(std::string& wgtId, int weightEntry) const;
    const WeightMetaInfo& weightMetaInfoByGlobalIndex(int weightEntry) const;
    int weightVectorEntry(std::string& wgtId) const;
    bool containsWeight(std::string& wgtId, int weightEntry) const;
    bool containsWeight(int weightEntry) const;
    int weightVectorEntry(std::string& wgtId, int weightEntry) const;
    void addContainedId(int weightEntry, std::string id, std::string label);
    bool indexInRange(int index) const;

    void setName(std::string name) { name_ = name; }
    void setDescription(std::string description) { description_ = description; }
    void appendDescription(std::string description) { description_ += description; }
    void setHeaderEntry(std::string header) { headerEntry_ = header; }
    void setWeightType(WeightType type) { weightType_ = type; }
    void setFirstId(int firstId) { firstId_ = firstId; }
    void setLastId(int lastId) { lastId_ = lastId; }
    // Call before doing lots of searches by label
    void cacheWeightIndicesByLabel();

    std::string name() const { return name_; }
    std::string description() const { return description_; }
    std::string headerEntry() const { return headerEntry_; }
    WeightType weightType() const { return weightType_; }
    std::vector<WeightMetaInfo> idsContained() const { return idsContained_; }
    size_t nIdsContained() const { return idsContained_.size(); }
    int firstId() const { return firstId_; }
    int lastId() const { return lastId_; }
    // Store whether the group was fully parsed succesfully
    void setIsWellFormed(bool wellFormed) { isWellFormed_ = wellFormed; }
    bool isWellFormed() const { return isWellFormed_; }
    int weightIndexFromLabel(std::string weightLabel) const;
    std::vector<std::string> weightLabels() const;

  protected:
    bool isWellFormed_ = false;
    std::string headerEntry_;
    std::string name_;
    std::string description_;
    WeightType weightType_;
    std::vector<WeightMetaInfo> idsContained_;
    int firstId_;
    int lastId_;
    std::unordered_map<std::string, size_t> weightLabelsToIndices_;
  };
}  // namespace gen

#endif  // SimDataFormats_GeneratorProducts_WeightGroupInfo_h

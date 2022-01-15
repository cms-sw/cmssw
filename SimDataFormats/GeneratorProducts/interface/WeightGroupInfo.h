#ifndef SimDataFormats_GeneratorProducts_WeightGroupInfo_h
#define SimDataFormats_GeneratorProducts_WeightGroupInfo_h

/** \class PdfInfo
 *
 */
#include <string>
#include <algorithm>
#include <vector>

namespace gen {
    struct WeightMetaInfo {
        size_t globalIndex;
        size_t localIndex;
        std::string id;
        std::string label;
    };

    enum WeightType {
        kPdfWeights,
        kScaleWeights,
        kMatrixElementWeights,
        kUnknownWeights,
        kShowerWeights,
    };

	class WeightGroupInfo {
        public:
            WeightGroupInfo() : WeightGroupInfo("") {}
	        WeightGroupInfo(std::string header, std::string name): 
                headerEntry_(header), name_(name), firstId_(-1), lastId_(-1) {}
	        WeightGroupInfo(std::string header): 
                headerEntry_(header), name_(header), firstId_(-1), lastId_(-1) {}
            WeightGroupInfo(const WeightGroupInfo &other) {
                copy(other);
            }
            WeightGroupInfo& operator=(const WeightGroupInfo &other) {
                copy(other);
                return *this; 
            }
            virtual ~WeightGroupInfo() {};
            void copy(const WeightGroupInfo &other);
            WeightGroupInfo* clone() const;
            WeightMetaInfo weightMetaInfo(int weightEntry);
            WeightMetaInfo weightMetaInfo(std::string wgtId);
            int weightVectorEntry(const std::string& wgtId);
            int containsWeight(const std::string& wgtId, int weightEntry);
            int weightVectorEntry(const std::string& wgtId, int weightEntry);
            void addContainedId(int weightEntry, std::string id, std::string label);
            std::vector<WeightMetaInfo> containedIds() const;
            bool indexInRange(int index) const;

            void setName(std::string name) { name_ = name; }
            void setHeaderEntry(std::string header) { headerEntry_ = header; }
            void setWeightType(WeightType type) { weightType_ = type; }
            void setFirstId(int firstId) { firstId_ = firstId; }
            void setLastId(int lastId) { lastId_ = lastId; }

            std::string name() const { return name_; }
            std::string headerEntry() const { return headerEntry_; }
            WeightType weightType() const { return weightType_; }
            std::vector<WeightMetaInfo> idsContained() const { return idsContained_; }
            int firstId() const { return firstId_; }
            int lastId() const { return lastId_; }

        protected:
            std::string headerEntry_;
            std::string name_;
            WeightType weightType_;
            std::vector<WeightMetaInfo> idsContained_;
            int firstId_;
            int lastId_;
	};
}

#endif // SimDataFormats_GeneratorProducts_WeightGroupInfo_h


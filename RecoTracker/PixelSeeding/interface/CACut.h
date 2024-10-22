#ifndef RecoTracker_PixelSeeding_CACut_h
#define RecoTracker_PixelSeeding_CACut_h
// -*- C++ -*-
// //
// // Package:    RecoTracker/PixelSeeding
// // Class:      CACut
// //
// // Original Author:  Karla Josefina Pena Rodriguez
// //         Created:  Wed, 14 Feb 2019 10:30:00 GMT
// //

#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "RecoTracker/PixelSeeding/interface/CAGraph.h"

class CACut {
public:
  explicit CACut(const double defaultCut, const std::vector<edm::ParameterSet> &tripletCuts)
      : useCACuts_(true), foundAllLayerIds_(false), defaultCut_(defaultCut) {
    if (tripletCuts.size() == 1 && tripletCuts[0].getParameter<double>("cut") == -1.) {
      useCACuts_ = false;
      LogDebug("Configuration") << "No CACut VPSet. Using default cut value of " << defaultCut
                                << " for all layer triplets";
      return;
    }

    valuesByTripletNames_.reserve(tripletCuts.size());
    valuesByLayerIds_.reserve(tripletCuts.size());

    setCutValuesByTripletNames(tripletCuts);
  }

  void setCutValuesByTripletNames(const std::vector<edm::ParameterSet> &tripletCuts) {
    for (const auto &thisTriplet : tripletCuts) {
      valuesByTripletNames_.emplace_back();
      auto &thisCACut = valuesByTripletNames_.back();

      thisCACut.tripletName_ = thisTriplet.getParameter<std::string>("seedingLayers");
      thisCACut.cutValue_ = thisTriplet.getParameter<double>("cut");
    }
  }

  void setCutValuesByLayerIds(CAGraph &caLayers) {
    if (!useCACuts_ || foundAllLayerIds_)
      return;

    foundAllLayerIds_ = true;
    valuesByLayerIds_.clear();

    for (const auto &thisTriplet : valuesByTripletNames_) {
      valuesByLayerIds_.emplace_back();
      auto &thisCACut = valuesByLayerIds_.back();

      // Triplet name, e.g. 'BPix1+BPix2+BPix3'
      std::string layersToSet = thisTriplet.tripletName_;
      for (int thisLayer = 0; thisLayer < 3; thisLayer++) {
        // Get layer name
        std::size_t layerPos = layersToSet.find('+');
        if ((thisLayer < 2 && layerPos == std::string::npos) || (thisLayer == 2 && layerPos != std::string::npos)) {
          throw cms::Exception("Configuration")
              << "Please enter a valid triplet name in the CACuts parameter set; e.g. 'BPix1+BPix2+BPix3'";
        }

        std::string layerName = layersToSet.substr(0, layerPos);
        layersToSet = layersToSet.substr(layerPos + 1);

        // Get layer ID
        thisCACut.layerIds_.emplace_back(caLayers.getLayerId(layerName));
        if (thisCACut.layerIds_.back() == -1) {
          foundAllLayerIds_ = false;
          edm::LogWarning("Configuration") << "Layer name '" << layerName
                                           << "' not found in the CAGraph. Please check CACuts parameter set if this "
                                              "warning is present for all events";
        }
      }

      // Cut
      thisCACut.cutValue_ = thisTriplet.cutValue_;
      thisCACut.hasValueByInnerLayerId_ = false;
    }

    setCutValuesByInnerLayerIds();
  }

  void setCutValuesByInnerLayerIds() {
    for (auto &thisTriplet : valuesByLayerIds_) {
      if (thisTriplet.hasValueByInnerLayerId_)
        continue;
      auto it = std::find(thisTriplet.layerIds_.begin(), thisTriplet.layerIds_.end(), -1);
      if (it != thisTriplet.layerIds_.end())
        continue;

      bool foundOuterDoublet = false;

      for (auto &thisOuterDoublet : valuesByInnerLayerIds_) {
        if (thisOuterDoublet.outerDoubletIds_[0] == thisTriplet.layerIds_[1] &&
            thisOuterDoublet.outerDoubletIds_[1] == thisTriplet.layerIds_[2]) {
          thisOuterDoublet.innerLayerIds_.emplace_back(thisTriplet.layerIds_[0]);
          thisOuterDoublet.cutValues_.emplace_back(thisTriplet.cutValue_);
          foundOuterDoublet = true;
          break;
        }
      }

      if (!foundOuterDoublet) {
        valuesByInnerLayerIds_.emplace_back(defaultCut_);
        auto &newOuterDoublet = valuesByInnerLayerIds_.back();

        newOuterDoublet.outerDoubletIds_.emplace_back(thisTriplet.layerIds_[1]);
        newOuterDoublet.outerDoubletIds_.emplace_back(thisTriplet.layerIds_[2]);
        newOuterDoublet.innerLayerIds_.emplace_back(thisTriplet.layerIds_[0]);
        newOuterDoublet.cutValues_.emplace_back(thisTriplet.cutValue_);
      }

      thisTriplet.hasValueByInnerLayerId_ = true;
    }
  }

  struct CAValuesByInnerLayerIds {
    explicit CAValuesByInnerLayerIds(float cut) : defaultCut_(cut) {}

    float at(int layerId) const {
      for (size_t thisLayer = 0; thisLayer < innerLayerIds_.size(); thisLayer++) {
        if (innerLayerIds_.at(thisLayer) == layerId)
          return cutValues_.at(thisLayer);
      }

      return defaultCut_;
    }

    std::vector<int> outerDoubletIds_;
    std::vector<int> innerLayerIds_;
    std::vector<float> cutValues_;

  private:
    double defaultCut_;
  };

  CAValuesByInnerLayerIds getCutsByInnerLayer(int layerIds1, int layerIds2) const {
    for (const auto &thisCut : valuesByInnerLayerIds_) {
      if (thisCut.outerDoubletIds_[0] == layerIds1 && thisCut.outerDoubletIds_[1] == layerIds2) {
        return thisCut;
      }
    }

    CAValuesByInnerLayerIds emptyCutsByInnerLayer(defaultCut_);
    return emptyCutsByInnerLayer;
  }

private:
  struct CAValueByTripletName {
    std::string tripletName_;
    float cutValue_;
  };

  struct CAValueByLayerIds {
    std::vector<int> layerIds_;
    float cutValue_;
    bool hasValueByInnerLayerId_;
  };

private:
  std::vector<CAValueByTripletName> valuesByTripletNames_;
  std::vector<CAValueByLayerIds> valuesByLayerIds_;
  std::vector<CAValuesByInnerLayerIds> valuesByInnerLayerIds_;
  bool useCACuts_;
  bool foundAllLayerIds_;
  const float defaultCut_;
};

#endif

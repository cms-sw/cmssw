#ifndef RecoPixelVertexing_PixelTriplets_interface_CAGraph_h
#define RecoPixelVertexing_PixelTriplets_interface_CAGraph_h

#include <array>
#include <string>
#include <vector>

struct CALayer {
  CALayer(const std::string &layerName, const int seqNum, std::size_t numberOfHits)
      : theName(layerName), theSeqNum(seqNum) {
    isOuterHitOfCell.resize(numberOfHits);
  }

  bool operator==(const std::string &otherString) const { return otherString == theName; }

  bool operator==(const int otherSeqNum) const { return otherSeqNum == theSeqNum; }

  const std::string &name() const { return theName; }

  const int seqNum() const { return theSeqNum; }

  std::vector<int> theOuterLayerPairs;
  std::vector<int> theInnerLayerPairs;

  std::vector<int> theOuterLayers;
  std::vector<int> theInnerLayers;
  std::vector<std::vector<unsigned int>> isOuterHitOfCell;

private:
  std::string theName;
  int theSeqNum;
};

struct CALayerPair {
  CALayerPair(int a, int b)

  {
    theLayers[0] = a;
    theLayers[1] = b;
  }

  bool operator==(const CALayerPair &otherLayerPair) {
    return (theLayers[0] == otherLayerPair.theLayers[0]) && (theLayers[1] == otherLayerPair.theLayers[1]);
  }

  std::array<int, 2> theLayers;
  std::array<unsigned int, 2> theFoundCells = {{0, 0}};
};

struct CAGraph {
  int getLayerId(const std::string &layerName) {
    for (const auto &thisLayer : theLayers) {
      if (thisLayer == layerName)
        return thisLayer.seqNum();
    }
    return -1;
  }

  std::vector<CALayer> theLayers;
  std::vector<CALayerPair> theLayerPairs;
  std::vector<int> theRootLayers;
};

#endif  // RecoPixelVertexing_PixelTriplets_interface_CAGraph_h

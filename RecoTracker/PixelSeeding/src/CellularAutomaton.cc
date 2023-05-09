#include <queue>

#include "CellularAutomaton.h"

void CellularAutomaton::createAndConnectCells(const std::vector<const HitDoublets *> &hitDoublets,
                                              const TrackingRegion &region,
                                              const CACut &thetaCut,
                                              const CACut &phiCut,
                                              const float hardPtCut) {
  int tsize = 0;
  for (auto hd : hitDoublets) {
    tsize += hd->size();
  }
  allCells.reserve(tsize);
  unsigned int cellId = 0;
  float ptmin = region.ptMin();
  float region_origin_x = region.origin().x();
  float region_origin_y = region.origin().y();
  float region_origin_radius = region.originRBound();

  std::vector<bool> alreadyVisitedLayerPairs;
  alreadyVisitedLayerPairs.resize(theLayerGraph.theLayerPairs.size());
  for (auto visited : alreadyVisitedLayerPairs) {
    visited = false;
  }
  for (int rootVertex : theLayerGraph.theRootLayers) {
    std::queue<int> LayerPairsToVisit;

    for (int LayerPair : theLayerGraph.theLayers[rootVertex].theOuterLayerPairs) {
      LayerPairsToVisit.push(LayerPair);
    }

    unsigned int numberOfLayerPairsToVisitAtThisDepth = LayerPairsToVisit.size();

    while (not LayerPairsToVisit.empty()) {
      auto currentLayerPair = LayerPairsToVisit.front();
      auto &currentLayerPairRef = theLayerGraph.theLayerPairs[currentLayerPair];
      auto &currentInnerLayerRef = theLayerGraph.theLayers[currentLayerPairRef.theLayers[0]];
      auto &currentOuterLayerRef = theLayerGraph.theLayers[currentLayerPairRef.theLayers[1]];
      bool allInnerLayerPairsAlreadyVisited{true};

      CACut::CAValuesByInnerLayerIds caThetaCut =
          thetaCut.getCutsByInnerLayer(currentInnerLayerRef.seqNum(), currentOuterLayerRef.seqNum());
      CACut::CAValuesByInnerLayerIds caPhiCut =
          phiCut.getCutsByInnerLayer(currentInnerLayerRef.seqNum(), currentOuterLayerRef.seqNum());

      for (auto innerLayerPair : currentInnerLayerRef.theInnerLayerPairs) {
        allInnerLayerPairsAlreadyVisited &= alreadyVisitedLayerPairs[innerLayerPair];
      }

      if (alreadyVisitedLayerPairs[currentLayerPair] == false and allInnerLayerPairsAlreadyVisited) {
        const HitDoublets *doubletLayerPairId = hitDoublets[currentLayerPair];
        auto numberOfDoublets = doubletLayerPairId->size();
        currentLayerPairRef.theFoundCells[0] = cellId;
        currentLayerPairRef.theFoundCells[1] = cellId + numberOfDoublets;
        for (unsigned int i = 0; i < numberOfDoublets; ++i) {
          allCells.emplace_back(
              doubletLayerPairId, i, doubletLayerPairId->innerHitId(i), doubletLayerPairId->outerHitId(i));

          currentOuterLayerRef.isOuterHitOfCell[doubletLayerPairId->outerHitId(i)].push_back(cellId);

          cellId++;

          auto &neigCells = currentInnerLayerRef.isOuterHitOfCell[doubletLayerPairId->innerHitId(i)];
          allCells.back().checkAlignmentAndTag(allCells,
                                               neigCells,
                                               ptmin,
                                               region_origin_x,
                                               region_origin_y,
                                               region_origin_radius,
                                               caThetaCut,
                                               caPhiCut,
                                               hardPtCut);
        }
        assert(cellId == currentLayerPairRef.theFoundCells[1]);
        for (auto outerLayerPair : currentOuterLayerRef.theOuterLayerPairs) {
          LayerPairsToVisit.push(outerLayerPair);
        }

        alreadyVisitedLayerPairs[currentLayerPair] = true;
      }
      LayerPairsToVisit.pop();
      numberOfLayerPairsToVisitAtThisDepth--;
      if (numberOfLayerPairsToVisitAtThisDepth == 0) {
        numberOfLayerPairsToVisitAtThisDepth = LayerPairsToVisit.size();
      }
    }
  }
}

void CellularAutomaton::evolve(const unsigned int minHitsPerNtuplet) {
  allStatus.resize(allCells.size());

  unsigned int numberOfIterations = minHitsPerNtuplet - 2;
  // keeping the last iteration for later
  for (unsigned int iteration = 0; iteration < numberOfIterations - 1; ++iteration) {
    for (auto &layerPair : theLayerGraph.theLayerPairs) {
      for (auto i = layerPair.theFoundCells[0]; i < layerPair.theFoundCells[1]; ++i) {
        allCells[i].evolve(i, allStatus);
      }
    }

    for (auto &layerPair : theLayerGraph.theLayerPairs) {
      for (auto i = layerPair.theFoundCells[0]; i < layerPair.theFoundCells[1]; ++i) {
        allStatus[i].updateState();
      }
    }
  }

  // last iteration

  for (int rootLayerId : theLayerGraph.theRootLayers) {
    for (int rootLayerPair : theLayerGraph.theLayers[rootLayerId].theOuterLayerPairs) {
      auto foundCells = theLayerGraph.theLayerPairs[rootLayerPair].theFoundCells;
      for (auto i = foundCells[0]; i < foundCells[1]; ++i) {
        auto &cell = allStatus[i];
        allCells[i].evolve(i, allStatus);
        cell.updateState();
        if (cell.isRootCell(minHitsPerNtuplet - 2)) {
          theRootCells.push_back(i);
        }
      }
    }
  }
}

void CellularAutomaton::findNtuplets(std::vector<CACell::CAntuplet> &foundNtuplets,
                                     const unsigned int minHitsPerNtuplet) {
  CACell::CAntuple tmpNtuplet;
  tmpNtuplet.reserve(minHitsPerNtuplet);

  for (auto root_cell : theRootCells) {
    tmpNtuplet.clear();
    tmpNtuplet.push_back(root_cell);
    allCells[root_cell].findNtuplets(allCells, foundNtuplets, tmpNtuplet, minHitsPerNtuplet);
  }
}

void CellularAutomaton::findTriplets(std::vector<const HitDoublets *> const &hitDoublets,
                                     std::vector<CACell::CAntuplet> &foundTriplets,
                                     TrackingRegion const &region,
                                     const CACut &thetaCut,
                                     const CACut &phiCut,
                                     const float hardPtCut) {
  int tsize = 0;
  for (auto hd : hitDoublets) {
    tsize += hd->size();
  }
  allCells.reserve(tsize);

  unsigned int cellId = 0;
  float ptmin = region.ptMin();
  float region_origin_x = region.origin().x();
  float region_origin_y = region.origin().y();
  float region_origin_radius = region.originRBound();

  std::vector<bool> alreadyVisitedLayerPairs;
  alreadyVisitedLayerPairs.resize(theLayerGraph.theLayerPairs.size());
  for (auto visited : alreadyVisitedLayerPairs) {
    visited = false;
  }
  for (int rootVertex : theLayerGraph.theRootLayers) {
    std::queue<int> LayerPairsToVisit;

    for (int LayerPair : theLayerGraph.theLayers[rootVertex].theOuterLayerPairs) {
      LayerPairsToVisit.push(LayerPair);
    }

    unsigned int numberOfLayerPairsToVisitAtThisDepth = LayerPairsToVisit.size();

    while (not LayerPairsToVisit.empty()) {
      auto currentLayerPair = LayerPairsToVisit.front();
      auto &currentLayerPairRef = theLayerGraph.theLayerPairs[currentLayerPair];
      auto &currentInnerLayerRef = theLayerGraph.theLayers[currentLayerPairRef.theLayers[0]];
      auto &currentOuterLayerRef = theLayerGraph.theLayers[currentLayerPairRef.theLayers[1]];
      bool allInnerLayerPairsAlreadyVisited{true};

      CACut::CAValuesByInnerLayerIds caThetaCut =
          thetaCut.getCutsByInnerLayer(currentInnerLayerRef.seqNum(), currentOuterLayerRef.seqNum());
      CACut::CAValuesByInnerLayerIds caPhiCut =
          phiCut.getCutsByInnerLayer(currentInnerLayerRef.seqNum(), currentOuterLayerRef.seqNum());

      for (auto innerLayerPair : currentInnerLayerRef.theInnerLayerPairs) {
        allInnerLayerPairsAlreadyVisited &= alreadyVisitedLayerPairs[innerLayerPair];
      }

      if (alreadyVisitedLayerPairs[currentLayerPair] == false and allInnerLayerPairsAlreadyVisited) {
        const HitDoublets *doubletLayerPairId = hitDoublets[currentLayerPair];
        auto numberOfDoublets = doubletLayerPairId->size();
        currentLayerPairRef.theFoundCells[0] = cellId;
        currentLayerPairRef.theFoundCells[1] = cellId + numberOfDoublets;
        for (unsigned int i = 0; i < numberOfDoublets; ++i) {
          allCells.emplace_back(
              doubletLayerPairId, i, doubletLayerPairId->innerHitId(i), doubletLayerPairId->outerHitId(i));

          currentOuterLayerRef.isOuterHitOfCell[doubletLayerPairId->outerHitId(i)].push_back(cellId);

          cellId++;

          auto &neigCells = currentInnerLayerRef.isOuterHitOfCell[doubletLayerPairId->innerHitId(i)];
          allCells.back().checkAlignmentAndPushTriplet(allCells,
                                                       neigCells,
                                                       foundTriplets,
                                                       ptmin,
                                                       region_origin_x,
                                                       region_origin_y,
                                                       region_origin_radius,
                                                       caThetaCut,
                                                       caPhiCut,
                                                       hardPtCut);
        }
        assert(cellId == currentLayerPairRef.theFoundCells[1]);
        for (auto outerLayerPair : currentOuterLayerRef.theOuterLayerPairs) {
          LayerPairsToVisit.push(outerLayerPair);
        }

        alreadyVisitedLayerPairs[currentLayerPair] = true;
      }
      LayerPairsToVisit.pop();
      numberOfLayerPairsToVisitAtThisDepth--;
      if (numberOfLayerPairsToVisitAtThisDepth == 0) {
        numberOfLayerPairsToVisitAtThisDepth = LayerPairsToVisit.size();
      }
    }
  }
}

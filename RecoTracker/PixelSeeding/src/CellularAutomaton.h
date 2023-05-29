#ifndef RecoTracker_PixelSeeding_src_CellularAutomaton_h
#define RecoTracker_PixelSeeding_src_CellularAutomaton_h

#include <array>

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "TrackingTools/TransientTrackingRecHit/interface/SeedingLayerSetsHits.h"

#include "CACell.h"
#include "RecoTracker/PixelSeeding/interface/CAGraph.h"

class CellularAutomaton {
public:
  CellularAutomaton(CAGraph& graph) : theLayerGraph(graph) {}

  std::vector<CACell>& getAllCells() { return allCells; }

  void createAndConnectCells(
      const std::vector<const HitDoublets*>&, const TrackingRegion&, const CACut&, const CACut&, const float);

  void evolve(const unsigned int);
  void findNtuplets(std::vector<CACell::CAntuplet>&, const unsigned int);
  void findTriplets(const std::vector<const HitDoublets*>& hitDoublets,
                    std::vector<CACell::CAntuplet>& foundTriplets,
                    const TrackingRegion& region,
                    const CACut& thetaCut,
                    const CACut& phiCut,
                    const float hardPtCut);

private:
  CAGraph& theLayerGraph;

  std::vector<CACell> allCells;
  std::vector<CACellStatus> allStatus;

  std::vector<unsigned int> theRootCells;
  std::vector<std::vector<CACell*> > theNtuplets;
};

#endif  // RecoTracker_PixelSeeding_src_CellularAutomaton_h

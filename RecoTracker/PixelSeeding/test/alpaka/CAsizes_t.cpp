#include <iostream>
#include <typeinfo>

#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/PixelSeeding/plugins/alpaka/CACell.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;

template <typename T>
void print() {
  std::cout << "size of " << typeid(T).name() << ' ' << sizeof(T) << std::endl;
}

int main() {
  using namespace pixelTopology;
  using namespace caStructures;

  //for Phase-I
  print<CACellT<Phase1>>();
  print<CellNeighborsT<Phase1>>();
  print<CellTracksT<Phase1>>();
  print<OuterHitOfCellContainerT<Phase1>>();
  print<TuplesContainerT<Phase1>>();
  print<HitToTupleT<Phase1>>();
  print<TupleMultiplicityT<Phase1>>();
  print<CellNeighborsVectorT<Phase1>>();

  //for Phase-II
  print<CACellT<Phase2>>();
  print<CellNeighborsT<Phase2>>();
  print<CellTracksT<Phase2>>();
  print<OuterHitOfCellContainerT<Phase2>>();
  print<TuplesContainerT<Phase2>>();
  print<HitToTupleT<Phase2>>();
  print<TupleMultiplicityT<Phase2>>();
  print<CellNeighborsVectorT<Phase2>>();

  return 0;
}

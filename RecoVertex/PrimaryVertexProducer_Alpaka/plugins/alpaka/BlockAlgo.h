#ifndef RecoVertex_PrimaryVertexProducer_Alpaka_plugins_alpaka_BlockAlgo_h
#define RecoVertex_PrimaryVertexProducer_Alpaka_plugins_alpaka_BlockAlgo_h

#include "DataFormats/PortableVertex/interface/alpaka/VertexDeviceCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class BlockAlgo {
  public:
    BlockAlgo();
    void createBlocks(Queue& queue,
                      const portablevertex::TrackDeviceCollection& inputTrack,
                      portablevertex::TrackDeviceCollection& trackInBlocks,
                      int32_t blockSize,
                      double blockOverlap);  // The actual block creation

  private:
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoVertex_PrimaryVertexProducer_Alpaka_plugins_alpaka_BlockAlgo_h

#ifndef RecoVertex_PrimaryVertexProducerplugins_alpaka_TracksForDAInBlocksAlgo_h
#define RecoVertex_PrimaryVertexProducerplugins_alpaka_TracksForDAInBlocksAlgo_h

#include "DataFormats/VertexSoA/interface/alpaka/VertexDeviceCollection.h"
#include "DataFormats/VertexSoA/interface/alpaka/TrackForVertexDeviceCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class TracksForDAInBlocksAlgo {
  public:
    TracksForDAInBlocksAlgo();
    void createBlocks(Queue& queue,
                      const reco::TrackForVertexDeviceCollection& inputTrack,
                      reco::TrackForVertexDeviceCollection& trackInBlocks,
                      int32_t blockSize,
                      double blockOverlap);  // The actual block creation

  private:
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoVertex_PrimaryVertexProducerplugins_alpaka_TracksForDAInBlocksAlgo_h

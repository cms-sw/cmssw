#ifndef RecoVertex_PrimaryVertexProducer_plugins_alpaka_DAInBlocksClusterizerAlgo_h
#define RecoVertex_PrimaryVertexProducer_plugins_alpaka_DAInBlocksClusterizerAlgo_h

// This header declares the host-callable public API of DAInBlocksClusterizerAlgo only, so it is
// safe to include from plain host-compiled wrapper .cc files (e.g. PrimaryVertexProducerPortable.cc).
// The device-only kernel-body implementation lives in DAInBlocksClusterizerAlgoKernels.h, which must
// only be included from .dev.cc translation units (see that header for why).

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "DataFormats/VertexSoA/interface/alpaka/VertexDeviceCollection.h"
#include "DataFormats/VertexSoA/interface/alpaka/TrackForVertexDeviceCollection.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  using namespace cms::alpakatools;

  struct DAInBlocksClusterParameters {
    double Tmin;
    double Tpurge;
    double Tstop;
    double vertexSize;
    double coolingFactor;
    double d0CutOff;
    double dzCutOff;
    double uniquetrkweight;
    double uniquetrkminp;
    double zmerge;
    double zrange;
    int32_t convergence_mode;
    double delta_lowT;
    double delta_highT;
  };

  class DAInBlocksClusterizerAlgo {
  public:
    DAInBlocksClusterizerAlgo(Queue& queue, int32_t bSize);

    void clusterize(Queue& queue,
                    reco::TrackForVertexDeviceCollection& inputTracks,
                    reco::VertexDeviceCollection& deviceVertex,
                    DAInBlocksClusterParameters const cParams,
                    int32_t nBlocks,
                    int32_t blockSize);  // Clusterization

    void resplit_tracks(Queue& queue,
                        reco::TrackForVertexDeviceCollection& inputTracks,
                        reco::VertexDeviceCollection& deviceVertex,
                        DAInBlocksClusterParameters const cParams,
                        int32_t nBlocks,
                        int32_t blockSize);  // Clusterization

    void reject_outliers(Queue& queue,
                         reco::TrackForVertexDeviceCollection& inputTracks,
                         reco::VertexDeviceCollection& deviceVertex,
                         DAInBlocksClusterParameters const cParams,
                         int32_t nBlocks,
                         int32_t blockSize);  // Clusterization
    void arbitrate(Queue& queue,
                   reco::TrackForVertexDeviceCollection& inputTracks,
                   reco::VertexDeviceCollection& deviceVertex,
                   DAInBlocksClusterParameters const cParams,
                   int32_t nBlocks,
                   int32_t blockSize);  // Arbitration

  private:
    cms::alpakatools::device_buffer<Device, double[]> beta_;
    cms::alpakatools::device_buffer<Device, double[]> osumtkwt_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoVertex_PrimaryVertexProducer_plugins_alpaka_DAInBlocksClusterizerAlgo_h

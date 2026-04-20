#ifndef RecoVertex_PrimaryVertexProducer_plugins_alpaka_DAInBlocksClusterizerAlgo_h
#define RecoVertex_PrimaryVertexProducer_plugins_alpaka_DAInBlocksClusterizerAlgo_h

#include "DataFormats/OfflineVertexSoA/interface/alpaka/VertexDeviceCollection.h"
#include "DataFormats/OfflineVertexSoA/interface/alpaka/TrackDeviceCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  struct ClusterParameters {
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
                    TrackDeviceCollection& inputTracks,
                    VertexDeviceCollection& deviceVertex,
                    ClusterParameters const& cParams,
                    int32_t nBlocks,
                    int32_t blockSize);  // Clusterization

    void resplit_tracks(Queue& queue,
                        TrackDeviceCollection& inputTracks,
                        VertexDeviceCollection& deviceVertex,
                        ClusterParameters const& cParams,
                        int32_t nBlocks,
                        int32_t blockSize);  // Clusterization

    void reject_outliers(Queue& queue,
                         TrackDeviceCollection& inputTracks,
                         VertexDeviceCollection& deviceVertex,
                         ClusterParameters const& cParams,
                         int32_t nBlocks,
                         int32_t blockSize);  // Clusterization
    void arbitrate(Queue& queue,
                   TrackDeviceCollection& inputTracks,
                   VertexDeviceCollection& deviceVertex,
                   ClusterParameters const& cParams,
                   int32_t nBlocks,
                   int32_t blockSize);  // Arbitration

  private:
    cms::alpakatools::device_buffer<Device, double[]> beta_;
    cms::alpakatools::device_buffer<Device, double[]> osumtkwt_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoVertex_PrimaryVertexProducer_plugins_alpaka_DAInBlocksClusterizerAlgo_h

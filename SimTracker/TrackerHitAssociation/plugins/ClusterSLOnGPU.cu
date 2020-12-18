#include <atomic>
#include <limits>
#include <mutex>

#include "CUDADataFormats/SiPixelCluster/interface/gpuClusteringConstants.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cuda_assert.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudastdAlgorithm.h"
#include "RecoLocalTracker/SiPixelClusterizer/plugins/SiPixelRawToClusterGPUKernel.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/pixelCPEforGPU.h"

#include "ClusterSLOnGPU.h"

using ClusterSLView = trackerHitAssociationHeterogeneous::ClusterSLView;
using Clus2TP = ClusterSLView::Clus2TP;

// #define DUMP_TK2

__global__ void simLink(const SiPixelDigisCUDA::DeviceConstView* dd,
                        uint32_t ndigis,
                        TrackingRecHit2DSOAView const* hhp,
                        ClusterSLView sl,
                        uint32_t n) {
  constexpr uint32_t invTK = 0;  // std::numeric_limits<int32_t>::max();
  using gpuClustering::invalidModuleId;
  using gpuClustering::maxNumModules;

  auto const& hh = *hhp;
  auto i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= ndigis)
    return;

  auto id = dd->moduleInd(i);
  if (invalidModuleId == id)
    return;
  assert(id < maxNumModules);

  auto ch = pixelgpudetails::pixelToChannel(dd->xx(i), dd->yy(i));
  auto first = hh.hitsModuleStart(id);
  auto cl = first + dd->clus(i);
  assert(cl < maxNumModules * blockDim.x);

  const Clus2TP me{{id, ch, 0, 0, 0, 0, 0}};

  auto less = [] __host__ __device__(Clus2TP const& a, Clus2TP const& b) -> bool {
    // in this context we do not care of [2]
    return a[0] < b[0] or ((not(b[0] < a[0])) and (a[1] < b[1]));
  };

  auto equal = [] __host__ __device__(Clus2TP const& a, Clus2TP const& b) -> bool {
    // in this context we do not care of [2]
    return a[0] == b[0] and a[1] == b[1];
  };

  auto const* b = sl.links_d;
  auto const* e = b + n;

  auto p = cuda_std::lower_bound(b, e, me, less);
  int32_t j = p - sl.links_d;
  assert(j >= 0);

  auto getTK = [&](int i) {
    auto const& l = sl.links_d[i];
    return l[2];
  };

  j = std::min(int(j), int(n - 1));
  if (equal(me, sl.links_d[j])) {
    auto const itk = j;
    auto const tk = getTK(j);
    auto old = atomicCAS(&sl.tkId_d[cl], invTK, itk);
    if (invTK == old or tk == getTK(old)) {
      atomicAdd(&sl.n1_d[cl], 1);
    } else {
      auto old = atomicCAS(&sl.tkId2_d[cl], invTK, itk);
      if (invTK == old or tk == getTK(old))
        atomicAdd(&sl.n2_d[cl], 1);
    }
  }
}

__global__ void doZero(uint32_t nhits, ClusterSLView sl) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i > nhits)
    return;

  sl.tkId_d[i] = 0;
  sl.n1_d[i] = 0;
  sl.tkId2_d[i] = 0;
  sl.n2_d[i] = 0;
}

__global__ void dumpLink(int first, int ev, TrackingRecHit2DSOAView const* hhp, uint32_t nhits, ClusterSLView sl) {
  auto i = first + blockIdx.x * blockDim.x + threadIdx.x;
  if (i > nhits)
    return;

  auto const& hh = *hhp;

  auto const& tk1 = sl.links_d[sl.tkId_d[i]];

#ifdef DUMP_TK2
  auto const& tk2 = sl.links_d[sl.tkId2_d[i]];

  printf("HIT: %d %d %d %d %.4f %.4f %.4f %.4f %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n",
#else
  printf("HIT: %d %d %d %d %.4f %.4f %.4f %.4f %d %d %d %d %d %d %d %d %d\n",
#endif
         ev,
         i,
         hh.detectorIndex(i),
         hh.charge(i),
         hh.xGlobal(i),
         hh.yGlobal(i),
         hh.zGlobal(i),
         hh.rGlobal(i),
         hh.iphi(i),
         hh.clusterSizeX(i),
         hh.clusterSizeY(i),
         tk1[2],
         tk1[3],
         tk1[4],
         tk1[5],
         tk1[6],
         sl.n1_d[i]
#ifdef DUMP_TK2
         ,
         tk2[2],
         tk2[3],
         tk2[4],
         tk2[5],
         tk2[6],
         sl.n2_d[i]
#endif
  );
}

namespace clusterSLOnGPU {

  void printCSVHeader() {
#ifdef DUMP_TK2
    printf("HIT: %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s\n",
#else
    printf("HIT: %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s\n",
#endif
           "ev",
           "ind",
           "det",
           "charge",
           "xg",
           "yg",
           "zg",
           "rg",
           "iphi",
           "xsize",
           "ysize",
           "tkId",
           "pt",
           "eta",
           "z0",
           "r0",
           "n1"
#ifdef DUMP_TK2
           ,
           "tkId2",
           "pt2",
           "eta",
           "z02",
           "r02",
           "n2"
#endif
    );
  }

  std::atomic<int> evId(0);
  std::once_flag doneCSVHeader;

  Kernel::Kernel(bool dump) : doDump(dump) {
    if (doDump)
      std::call_once(doneCSVHeader, printCSVHeader);
  }

  trackerHitAssociationHeterogeneous::ProductCUDA Kernel::makeAsync(SiPixelDigisCUDA const& dd,
                                                                    uint32_t ndigis,
                                                                    HitsOnCPU const& hh,
                                                                    Clus2TP const* digi2tp,
                                                                    uint32_t nhits,
                                                                    uint32_t nlinks,
                                                                    cudaStream_t stream) const {
    trackerHitAssociationHeterogeneous::ProductCUDA product(nlinks, nhits, stream);
    auto& csl = product.view();

    cudaCheck(cudaMemcpyAsync(csl.links_d, digi2tp, sizeof(Clus2TP) * nlinks, cudaMemcpyDefault, stream));

    if (0 == nhits)
      return product;

    int ev = ++evId;
    int threadsPerBlock = 256;

    int blocks = (nhits + threadsPerBlock - 1) / threadsPerBlock;
    doZero<<<blocks, threadsPerBlock, 0, stream>>>(nhits, csl);
    cudaCheck(cudaGetLastError());

    blocks = (ndigis + threadsPerBlock - 1) / threadsPerBlock;
    simLink<<<blocks, threadsPerBlock, 0, stream>>>(dd.view(), ndigis, hh.view(), csl, nlinks);
    cudaCheck(cudaGetLastError());

    if (doDump) {
      cudaStreamSynchronize(stream);  // flush previous printf
      // one line == 200B so each kernel can print only 5K lines....
      blocks = 16;
      for (int first = 0; first < int(nhits); first += blocks * threadsPerBlock) {
        dumpLink<<<blocks, threadsPerBlock, 0, stream>>>(first, ev, hh.view(), nhits, csl);
        cudaCheck(cudaGetLastError());
        cudaStreamSynchronize(stream);
      }
    }
    cudaCheck(cudaGetLastError());

    return product;
  }

}  // namespace clusterSLOnGPU

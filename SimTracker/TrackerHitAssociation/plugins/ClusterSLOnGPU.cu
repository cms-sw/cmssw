#include <atomic>
#include <limits>
#include <mutex>

#include "HeterogeneousCore/CUDAUtilities/interface/cuda_assert.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudastdAlgorithm.h"
#include "RecoLocalTracker/SiPixelClusterizer/plugins/SiPixelRawToClusterGPUKernel.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/pixelCPEforGPU.h"

#include "ClusterSLOnGPU.h"

using ClusterSLGPU = trackerHitAssociationHeterogeneousProduct::ClusterSLGPU;
using Clus2TP = ClusterSLGPU::Clus2TP;

// #define DUMP_TK2

__global__ void simLink(const SiPixelDigisCUDA::DeviceConstView* dd,
                        uint32_t ndigis,
                        clusterSLOnGPU::HitsOnGPU const* hhp,
                        ClusterSLGPU const* slp,
                        uint32_t n) {
  assert(slp == slp->me_d);

  constexpr int32_t invTK = 0;      // std::numeric_limits<int32_t>::max();
  constexpr uint16_t InvId = 9999;  // must be > MaxNumModules

  auto const& hh = *hhp;
  auto const& sl = *slp;
  auto i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= ndigis)
    return;

  auto id = dd->moduleInd(i);
  if (InvId == id)
    return;
  assert(id < 2000);

  auto ch = pixelgpudetails::pixelToChannel(dd->xx(i), dd->yy(i));
  auto first = hh.hitsModuleStart(id);
  auto cl = first + dd->clus(i);
  assert(cl < 2000 * blockDim.x);

  const Clus2TP me{{id, ch, 0, 0, 0, 0}};

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

__global__ void verifyZero(uint32_t nhits, ClusterSLGPU const* slp) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i > nhits)
    return;

  auto const& sl = *slp;

  assert(sl.tkId_d[i] == 0);
  auto const& tk = sl.links_d[0];
  assert(tk[0] == 0);
  assert(tk[1] == 0);
  assert(tk[2] == 0);
  assert(tk[3] == 0);
}

__global__ void dumpLink(
    int first, int ev, clusterSLOnGPU::HitsOnGPU const* hhp, uint32_t nhits, ClusterSLGPU const* slp) {
  auto i = first + blockIdx.x * blockDim.x + threadIdx.x;
  if (i > nhits)
    return;

  auto const& hh = *hhp;
  auto const& sl = *slp;

  auto const& tk1 = sl.links_d[sl.tkId_d[i]];

#ifdef DUMP_TK2
  auto const& tk2 = sl.links_d[sl.tkId2_d[i]];

  printf("HIT: %d %d %d %d %.4f %.4f %.4f %.4f %d %d %d %d %d %d %d %d %d %d %d %d %d\n",
#else
  printf("HIT: %d %d %d %d %.4f %.4f %.4f %.4f %d %d %d %d %d %d %d %d\n",
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
         sl.n1_d[i]
#ifdef DUMP_TK2
         ,
         tk2[2],
         tk2[3],
         tk2[4],
         tk2[5],
         sl.n2_d[i]
#endif
  );
}

namespace clusterSLOnGPU {

  constexpr uint32_t invTK = 0;  // std::numeric_limits<int32_t>::max();

  void printCSVHeader() {
#ifdef DUMP_TK2
    printf("HIT: %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s\n",
#else
    printf("HIT: %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s\n",
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
           "z0",
           "r0",
           "n1"
#ifdef DUMP_TK2
           ,
           "tkId2",
           "pt2",
           "z02",
           "r02",
           "n2"
#endif
    );
  }

  std::atomic<int> evId(0);
  std::once_flag doneCSVHeader;

  Kernel::Kernel(cudaStream_t stream, bool dump) : doDump(dump) {
    if (doDump)
      std::call_once(doneCSVHeader, printCSVHeader);
    alloc(stream);
  }

  void Kernel::alloc(cudaStream_t stream) {
    cudaCheck(cudaMalloc((void**)&slgpu.links_d, (ClusterSLGPU::MAX_DIGIS) * sizeof(Clus2TP)));
    cudaCheck(cudaMalloc((void**)&slgpu.tkId_d, (ClusterSLGPU::MaxNumModules * 256) * sizeof(uint32_t)));
    cudaCheck(cudaMalloc((void**)&slgpu.tkId2_d, (ClusterSLGPU::MaxNumModules * 256) * sizeof(uint32_t)));
    cudaCheck(cudaMalloc((void**)&slgpu.n1_d, (ClusterSLGPU::MaxNumModules * 256) * sizeof(uint32_t)));
    cudaCheck(cudaMalloc((void**)&slgpu.n2_d, (ClusterSLGPU::MaxNumModules * 256) * sizeof(uint32_t)));
    cudaCheck(cudaMalloc((void**)&slgpu.me_d, sizeof(ClusterSLGPU)));
    cudaCheck(cudaMemcpyAsync(slgpu.me_d, &slgpu, sizeof(ClusterSLGPU), cudaMemcpyDefault, stream));
  }

  void Kernel::deAlloc() {
    cudaCheck(cudaFree(slgpu.links_d));
    cudaCheck(cudaFree(slgpu.tkId_d));
    cudaCheck(cudaFree(slgpu.tkId2_d));
    cudaCheck(cudaFree(slgpu.n1_d));
    cudaCheck(cudaFree(slgpu.n2_d));
    cudaCheck(cudaFree(slgpu.me_d));
  }

  void Kernel::zero(cudaStream_t stream) {
    cudaCheck(cudaMemsetAsync(slgpu.tkId_d, invTK, (ClusterSLGPU::MaxNumModules * 256) * sizeof(uint32_t), stream));
    cudaCheck(cudaMemsetAsync(slgpu.tkId2_d, invTK, (ClusterSLGPU::MaxNumModules * 256) * sizeof(uint32_t), stream));
    cudaCheck(cudaMemsetAsync(slgpu.n1_d, 0, (ClusterSLGPU::MaxNumModules * 256) * sizeof(uint32_t), stream));
    cudaCheck(cudaMemsetAsync(slgpu.n2_d, 0, (ClusterSLGPU::MaxNumModules * 256) * sizeof(uint32_t), stream));
  }

  void Kernel::algo(SiPixelDigisCUDA const& dd,
                    uint32_t ndigis,
                    HitsOnCPU const& hh,
                    uint32_t nhits,
                    uint32_t n,
                    cudaStream_t stream) {
    zero(stream);

    if (0 == nhits)
      return;
    ClusterSLGPU const& sl = slgpu;

    int ev = ++evId;
    int threadsPerBlock = 256;

    int blocks = (nhits + threadsPerBlock - 1) / threadsPerBlock;
    verifyZero<<<blocks, threadsPerBlock, 0, stream>>>(nhits, sl.me_d);
    cudaCheck(cudaGetLastError());

    blocks = (ndigis + threadsPerBlock - 1) / threadsPerBlock;

    assert(sl.me_d);
    simLink<<<blocks, threadsPerBlock, 0, stream>>>(dd.view(), ndigis, hh.view(), sl.me_d, n);
    cudaCheck(cudaGetLastError());

    if (doDump) {
      cudaStreamSynchronize(stream);  // flush previous printf
      // one line == 200B so each kernel can print only 5K lines....
      blocks = 16;  // (nhits + threadsPerBlock - 1) / threadsPerBlock;
      for (int first = 0; first < int(nhits); first += blocks * threadsPerBlock) {
        dumpLink<<<blocks, threadsPerBlock, 0, stream>>>(first, ev, hh.view(), nhits, sl.me_d);
        cudaCheck(cudaGetLastError());
        cudaStreamSynchronize(stream);
      }
    }
    cudaCheck(cudaGetLastError());
  }

}  // namespace clusterSLOnGPU

#include <atomic>
#include <limits>
#include <mutex>

#include "HeterogeneousCore/CUDAUtilities/interface/cuda_assert.h"
#include "CUDADataFormats/SiPixelDigi/interface/SiPixelDigisCUDA.h"
#include "CUDADataFormats/SiPixelCluster/interface/SiPixelClustersCUDA.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudastdAlgorithm.h"
#include "RecoLocalTracker/SiPixelClusterizer/plugins/SiPixelRawToClusterGPUKernel.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/pixelCPEforGPU.h"

#include "ClusterSLOnGPU.h"

using ClusterSLGPU = trackerHitAssociationHeterogeneousProduct::ClusterSLGPU;

__global__
void simLink(const SiPixelDigisCUDA::DeviceConstView *dd, uint32_t ndigis, const SiPixelClustersCUDA::DeviceConstView *cc, clusterSLOnGPU::HitsOnGPU const * hhp, ClusterSLGPU const * slp, uint32_t n)
{
  assert(slp == slp->me_d);

  constexpr int32_t  invTK = 0;     // std::numeric_limits<int32_t>::max();
  constexpr uint16_t InvId = 9999;  // must be > MaxNumModules

  auto const & hh = *hhp;
  auto const & sl = *slp;
  auto i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= ndigis)
    return;

  auto id = dd->moduleInd(i);
  if (InvId == id)
    return;
  assert(id < 2000);

  auto ch = pixelgpudetails::pixelToChannel(dd->xx(i), dd->yy(i));
  auto first = hh.hitsModuleStart_d[id];
  auto cl = first + cc->clus(i);
  assert(cl < 2000 * blockDim.x);

  const std::array<uint32_t, 4> me{{id, ch, 0, 0}};

  auto less = [] __host__ __device__ (std::array<uint32_t, 4> const & a, std::array<uint32_t, 4> const & b)->bool {
     // in this context we do not care of [2]
     return a[0] < b[0] or (not b[0] < a[0] and a[1] < b[1]);
  };

  auto equal = [] __host__ __device__ (std::array<uint32_t, 4> const & a, std::array<uint32_t, 4> const & b)->bool {
     // in this context we do not care of [2]
     return a[0] == b[0] and a[1] == b[1];
  };

  auto const * b = sl.links_d;
  auto const * e = b + n;

  auto p = cuda_std::lower_bound(b, e, me, less);
  int32_t j = p-sl.links_d;
  assert(j >= 0);

  auto getTK = [&](int i) { auto const & l = sl.links_d[i]; return l[2];};

  j = std::min(int(j), int(n-1));
  if (equal(me, sl.links_d[j])) {
    auto const itk = j;
    auto const tk = getTK(j);
    auto old = atomicCAS(&sl.tkId_d[cl], invTK, itk);
    if (invTK == old or tk == getTK(old)) {
       atomicAdd(&sl.n1_d[cl], 1);
    } else {
      auto old = atomicCAS(&sl.tkId2_d[cl], invTK, itk);
      if (invTK == old or tk == getTK(old)) atomicAdd(&sl.n2_d[cl], 1);
    }
  }
}

__global__
void verifyZero(uint32_t nhits, ClusterSLGPU const * slp) {
  auto i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i > nhits)
    return;

  auto const & sl = *slp;

  assert(sl.tkId_d[i]==0);
  auto const & tk = sl.links_d[0];
  assert(tk[0]==0);
  assert(tk[1]==0);
  assert(tk[2]==0);
  assert(tk[3]==0);
}

__global__
void dumpLink(int first, int ev, clusterSLOnGPU::HitsOnGPU const * hhp, uint32_t nhits, ClusterSLGPU const * slp) {
  auto i = first + blockIdx.x*blockDim.x + threadIdx.x;
  if (i>nhits) return;

  auto const & hh = *hhp;
  auto const & sl = *slp;

  /* just an example of use of global error....
  assert(hh.cpeParams);
  float ge[6];
  hh.cpeParams->detParams(hh.detInd_d[i]).frame.toGlobal(hh.xerr_d[i], 0, hh.yerr_d[i],ge);
  printf("Error: %d: %f,%f,%f,%f,%f,%f\n",hh.detInd_d[i],ge[0],ge[1],ge[2],ge[3],ge[4],ge[5]);
  */

  auto const & tk1 = sl.links_d[sl.tkId_d[i]];
  auto const & tk2 = sl.links_d[sl.tkId2_d[i]];

  printf("HIT: %d %d %d %d %f %f %f %f %d %d %d %d %d %d %d\n", ev, i,
         hh.detInd_d[i], hh.charge_d[i],
         hh.xg_d[i], hh.yg_d[i], hh.zg_d[i], hh.rg_d[i], hh.iphi_d[i],
         tk1[2], tk1[3], sl.n1_d[i],
         tk2[2], tk2[3], sl.n2_d[i]
        );

}

namespace clusterSLOnGPU {

  constexpr uint32_t invTK = 0; // std::numeric_limits<int32_t>::max();

  void printCSVHeader() {
    printf("HIT: %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s\n", "ev", "ind",
        "det", "charge",
        "xg","yg","zg","rg","iphi",
        "tkId","pt","n1","tkId2","pt2","n2"
        );
  }

  std::atomic<int> evId(0);
  std::once_flag doneCSVHeader;

  Kernel::Kernel(cuda::stream_t<>& stream, bool dump) : doDump(dump) {
    if (doDump) std::call_once(doneCSVHeader, printCSVHeader);
    alloc(stream);
  }

  void Kernel::alloc(cuda::stream_t<>& stream) {
    cudaCheck(cudaMalloc((void**) & slgpu.links_d, (ClusterSLGPU::MAX_DIGIS)*sizeof(std::array<uint32_t, 4>)));
    cudaCheck(cudaMalloc((void**) & slgpu.tkId_d, (ClusterSLGPU::MaxNumModules*256)*sizeof(uint32_t)));
    cudaCheck(cudaMalloc((void**) & slgpu.tkId2_d, (ClusterSLGPU::MaxNumModules*256)*sizeof(uint32_t)));
    cudaCheck(cudaMalloc((void**) & slgpu.n1_d, (ClusterSLGPU::MaxNumModules*256)*sizeof(uint32_t)));
    cudaCheck(cudaMalloc((void**) & slgpu.n2_d, (ClusterSLGPU::MaxNumModules*256)*sizeof(uint32_t)));
    cudaCheck(cudaMalloc((void**) & slgpu.me_d, sizeof(ClusterSLGPU)));
    cudaCheck(cudaMemcpyAsync(slgpu.me_d, &slgpu, sizeof(ClusterSLGPU), cudaMemcpyDefault, stream.id()));
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
    cudaCheck(cudaMemsetAsync(slgpu.tkId_d, invTK, (ClusterSLGPU::MaxNumModules*256)*sizeof(uint32_t), stream));
    cudaCheck(cudaMemsetAsync(slgpu.tkId2_d, invTK, (ClusterSLGPU::MaxNumModules*256)*sizeof(uint32_t), stream));
    cudaCheck(cudaMemsetAsync(slgpu.n1_d, 0, (ClusterSLGPU::MaxNumModules*256)*sizeof(uint32_t), stream));
    cudaCheck(cudaMemsetAsync(slgpu.n2_d, 0, (ClusterSLGPU::MaxNumModules*256)*sizeof(uint32_t), stream));
  }

  void Kernel::algo(DigisOnGPU const & dd, uint32_t ndigis, HitsOnCPU const & hh, uint32_t nhits, uint32_t n, cuda::stream_t<>& stream) {
    zero(stream.id());

    ClusterSLGPU const & sl = slgpu;

    int ev = ++evId;
    int threadsPerBlock = 256;

    int blocks = (nhits + threadsPerBlock - 1) / threadsPerBlock;
    verifyZero<<<blocks, threadsPerBlock, 0, stream.id()>>>(nhits, sl.me_d);
    cudaCheck(cudaGetLastError());

    blocks = (ndigis + threadsPerBlock - 1) / threadsPerBlock;

    assert(sl.me_d);
    simLink<<<blocks, threadsPerBlock, 0, stream.id()>>>(dd.digis_d.view(), ndigis, dd.clusters_d.view(), hh.gpu_d, sl.me_d, n);
    cudaCheck(cudaGetLastError());

    if (doDump) {
      cudaStreamSynchronize(stream.id());	// flush previous printf
      // one line == 200B so each kernel can print only 5K lines....
      blocks = 16; // (nhits + threadsPerBlock - 1) / threadsPerBlock;
      for (int first=0; first<int(nhits); first+=blocks*threadsPerBlock) {
        dumpLink<<<blocks, threadsPerBlock, 0, stream.id()>>>(first, ev, hh.gpu_d, nhits, sl.me_d);
        cudaCheck(cudaGetLastError());
        cudaStreamSynchronize(stream.id());
      }
    }
    cudaCheck(cudaGetLastError());
  }

}

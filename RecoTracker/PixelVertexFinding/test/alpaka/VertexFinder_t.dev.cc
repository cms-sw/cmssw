#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>
#include <alpaka/alpaka.hpp>
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
// TrackUtilities only included in order to compile SoALayout with Eigen columns
#include "DataFormats/TrackSoA/interface/alpaka/TrackUtilities.h"
#ifdef USE_DBSCAN
#include "RecoTracker/PixelVertexFinding/plugins/alpaka/clusterTracksDBSCAN.h"
#define CLUSTERIZE ALPAKA_ACCELERATOR_NAMESPACE::vertexFinder::ClusterTracksDBSCAN
#elif USE_ITERATIVE
#include "RecoTracker/PixelVertexFinding/plugins/alpaka/clusterTracksIterative.h"
#define CLUSTERIZE ALPAKA_ACCELERATOR_NAMESPACE::vertexFinder::ClusterTracksIterative
#else
#include "RecoTracker/PixelVertexFinding/plugins/alpaka/clusterTracksByDensity.h"
#define CLUSTERIZE ALPAKA_ACCELERATOR_NAMESPACE::vertexFinder::ClusterTracksByDensityKernel
#endif

#include "RecoTracker/PixelVertexFinding/interface/PixelVertexWorkSpaceLayout.h"
#include "RecoTracker/PixelVertexFinding/plugins/PixelVertexWorkSpaceSoAHostAlpaka.h"
#include "RecoTracker/PixelVertexFinding/plugins/alpaka/PixelVertexWorkSpaceSoADeviceAlpaka.h"

#include "RecoTracker/PixelVertexFinding/plugins/alpaka/fitVertices.h"
#include "RecoTracker/PixelVertexFinding/plugins/alpaka/sortByPt2.h"
#include "RecoTracker/PixelVertexFinding/plugins/alpaka/splitVertices.h"
#include "RecoTracker/PixelVertexFinding/plugins/alpaka/vertexFinder.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  using namespace cms::alpakatools;

  struct ClusterGenerator {
    explicit ClusterGenerator(float nvert, float ntrack)
        : rgen(-13., 13), errgen(0.005, 0.025), clusGen(nvert), trackGen(ntrack), gauss(0., 1.), ptGen(1.) {}

    void operator()(vertexFinder::PixelVertexWorkSpaceSoAHost& pwsh, ZVertexHost& vtxh) {
      int nclus = clusGen(reng);
      for (int zint = 0; zint < vtxh.view().metadata().size(); ++zint) {
        vtxh.view().zv()[zint] = 3.5f * gauss(reng);
      }

      int aux = 0;
      for (int iv = 0; iv < nclus; ++iv) {
        auto nt = trackGen(reng);
        pwsh.view().itrk()[iv] = nt;
        for (int it = 0; it < nt; ++it) {
          auto err = errgen(reng);  // reality is not flat....
          pwsh.view().zt()[aux] = vtxh.view().zv()[iv] + err * gauss(reng);
          pwsh.view().ezt2()[aux] = err * err;
          pwsh.view().iv()[aux] = iv;
          pwsh.view().ptt2()[aux] = (iv == 5 ? 1.f : 0.5f) + ptGen(reng);
          pwsh.view().ptt2()[aux] *= pwsh.view().ptt2()[aux];
          ++aux;
        }
      }
      pwsh.view().ntrks() = aux;
      // add noise
      auto nt = 2 * trackGen(reng);
      for (int it = 0; it < nt; ++it) {
        auto err = 0.03f;
        pwsh.view().zt()[it] = rgen(reng);
        pwsh.view().ezt2()[it] = err * err;
        pwsh.view().iv()[it] = 9999;
        pwsh.view().ptt2()[it] = 0.5f + ptGen(reng);
        pwsh.view().ptt2()[it] *= pwsh.view().ptt2()[it];
      }
    }

    std::mt19937 reng;
    std::uniform_real_distribution<float> rgen;
    std::uniform_real_distribution<float> errgen;
    std::poisson_distribution<int> clusGen;
    std::poisson_distribution<int> trackGen;
    std::normal_distribution<float> gauss;
    std::exponential_distribution<float> ptGen;
  };

  namespace vertexfinder_t {
#ifdef ONE_KERNEL
    class VertexFinderOneKernel {
    public:
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                    vertexFinder::VtxSoAView pdata,
                                    vertexFinder::WsSoAView pws,
                                    int minT,      // min number of neighbours to be "seed"
                                    float eps,     // max absolute distance to cluster
                                    float errmax,  // max error to be "seed"
                                    float chi2max  // max normalized distance to cluster,
      ) const {
        vertexFinder::clusterTracksByDensity(acc, pdata, pws, minT, eps, errmax, chi2max);
        alpaka::syncBlockThreads(acc);
        vertexFinder::fitVertices(acc, pdata, pws, 50.);
        alpaka::syncBlockThreads(acc);
        vertexFinder::splitVertices(acc, pdata, pws, 9.f);
        alpaka::syncBlockThreads(acc);
        vertexFinder::fitVertices(acc, pdata, pws, 5000.);
        alpaka::syncBlockThreads(acc);
        vertexFinder::sortByPt2(acc, pdata, pws);
        alpaka::syncBlockThreads(acc);
      }
    };
#endif

    class Kernel_print {
    public:
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                    vertexFinder::VtxSoAView pdata,
                                    vertexFinder::WsSoAView pws) const {
        printf("nt,nv %d %d,%d\n", pws.ntrks(), pdata.nvFinal(), pws.nvIntermediate());
      }
    };

    void runKernels(Queue& queue) {
      vertexFinder::PixelVertexWorkSpaceSoADevice ws_d(zVertex::MAXTRACKS, queue);
      vertexFinder::PixelVertexWorkSpaceSoAHost ws_h(zVertex::MAXTRACKS, queue);
      ZVertexHost vertices_h(queue);
      ZVertexSoACollection vertices_d(queue);

      float eps = 0.1f;
      std::array<float, 3> par{{eps, 0.01f, 9.0f}};
      for (int nav = 30; nav < 80; nav += 20) {
        ClusterGenerator gen(nav, 10);

        for (int i = 8; i < 20; ++i) {
          auto kk = i / 4;  // M param

          gen(ws_h, vertices_h);
          auto workDiv1D = make_workdiv<Acc1D>(1, 1);
          alpaka::exec<Acc1D>(queue, workDiv1D, vertexFinder::Init{}, vertices_d.view(), ws_d.view());
          // std::cout << "v,t size " << ws_h.view().zt()[0] << ' ' << vertices_h.view().zv()[0] << std::endl;
          alpaka::memcpy(queue, ws_d.buffer(), ws_h.buffer());
          alpaka::wait(queue);

          std::cout << "M eps, pset " << kk << ' ' << eps << ' ' << (i % 4) << std::endl;

          if ((i % 4) == 0)
            par = {{eps, 0.02f, 12.0f}};
          if ((i % 4) == 1)
            par = {{eps, 0.02f, 9.0f}};
          if ((i % 4) == 2)
            par = {{eps, 0.01f, 9.0f}};
          if ((i % 4) == 3)
            par = {{0.7f * eps, 0.01f, 9.0f}};

          alpaka::exec<Acc1D>(queue, workDiv1D, Kernel_print{}, vertices_d.view(), ws_d.view());

          auto workDivClusterizer = make_workdiv<Acc1D>(1, 512 + 256);
#ifdef ONE_KERNEL
          alpaka::exec<Acc1D>(queue,
                              workDivClusterizer,
                              VertexFinderOneKernel{},
                              vertices_d.view(),
                              ws_d.view(),
                              kk,
                              par[0],
                              par[1],
                              par[2]);
#else
          alpaka::exec<Acc1D>(
              queue, workDivClusterizer, CLUSTERIZE{}, vertices_d.view(), ws_d.view(), kk, par[0], par[1], par[2]);
#endif
          alpaka::wait(queue);
          alpaka::exec<Acc1D>(queue, workDiv1D, Kernel_print{}, vertices_d.view(), ws_d.view());
          alpaka::wait(queue);

          auto workDivFitter = make_workdiv<Acc1D>(1, 1024 - 256);

          alpaka::exec<Acc1D>(
              queue, workDivFitter, vertexFinder::FitVerticesKernel{}, vertices_d.view(), ws_d.view(), 50.f);

          alpaka::memcpy(queue, vertices_h.buffer(), vertices_d.buffer());
          alpaka::wait(queue);

          if (vertices_h.view().nvFinal() == 0) {
            std::cout << "NO VERTICES???" << std::endl;
            continue;
          }

          for (auto j = 0U; j < vertices_h.view().nvFinal(); ++j)
            if (vertices_h.view().ndof()[j] > 0)
              vertices_h.view().chi2()[j] /= float(vertices_h.view().ndof()[j]);
          {
            auto mx =
                std::minmax_element(vertices_h.view().chi2(), vertices_h.view().chi2() + vertices_h.view().nvFinal());
            std::cout << "after fit nv, min max chi2 " << vertices_h.view().nvFinal() << " " << *mx.first << ' '
                      << *mx.second << std::endl;
          }

          alpaka::exec<Acc1D>(
              queue, workDivFitter, vertexFinder::FitVerticesKernel{}, vertices_d.view(), ws_d.view(), 50.f);
          alpaka::memcpy(queue, vertices_h.buffer(), vertices_d.buffer());
          alpaka::wait(queue);

          for (auto j = 0U; j < vertices_h.view().nvFinal(); ++j)
            if (vertices_h.view().ndof()[j] > 0)
              vertices_h.view().chi2()[j] /= float(vertices_h.view().ndof()[j]);
          {
            auto mx =
                std::minmax_element(vertices_h.view().chi2(), vertices_h.view().chi2() + vertices_h.view().nvFinal());
            std::cout << "before splitting nv, min max chi2 " << vertices_h.view().nvFinal() << " " << *mx.first << ' '
                      << *mx.second << std::endl;
          }

          auto workDivSplitter = make_workdiv<Acc1D>(1024, 64);

          // one vertex per block!!!
          alpaka::exec<Acc1D>(
              queue, workDivSplitter, vertexFinder::SplitVerticesKernel{}, vertices_d.view(), ws_d.view(), 9.f);
          alpaka::memcpy(queue, ws_h.buffer(), ws_d.buffer());
          alpaka::wait(queue);
          std::cout << "after split " << ws_h.view().nvIntermediate() << std::endl;

          alpaka::exec<Acc1D>(
              queue, workDivFitter, vertexFinder::FitVerticesKernel{}, vertices_d.view(), ws_d.view(), 5000.f);

          auto workDivSorter = make_workdiv<Acc1D>(1, 256);
          alpaka::exec<Acc1D>(queue, workDivSorter, vertexFinder::SortByPt2Kernel{}, vertices_d.view(), ws_d.view());
          alpaka::memcpy(queue, vertices_h.buffer(), vertices_d.buffer());
          alpaka::wait(queue);

          if (vertices_h.view().nvFinal() == 0) {
            std::cout << "NO VERTICES???" << std::endl;
            continue;
          }

          for (auto j = 0U; j < vertices_h.view().nvFinal(); ++j)
            if (vertices_h.view().ndof()[j] > 0)
              vertices_h.view().chi2()[j] /= float(vertices_h.view().ndof()[j]);
          {
            auto mx =
                std::minmax_element(vertices_h.view().chi2(), vertices_h.view().chi2() + vertices_h.view().nvFinal());
            std::cout << "nv, min max chi2 " << vertices_h.view().nvFinal() << " " << *mx.first << ' ' << *mx.second
                      << std::endl;
          }

          {
            auto mx = std::minmax_element(vertices_h.view().wv(), vertices_h.view().wv() + vertices_h.view().nvFinal());
            std::cout << "min max error " << 1. / std::sqrt(*mx.first) << ' ' << 1. / std::sqrt(*mx.second)
                      << std::endl;
          }

          {
            auto mx =
                std::minmax_element(vertices_h.view().ptv2(), vertices_h.view().ptv2() + vertices_h.view().nvFinal());
            std::cout << "min max ptv2 " << *mx.first << ' ' << *mx.second << std::endl;
            std::cout << "min max ptv2 " << vertices_h.view().ptv2()[vertices_h.view().sortInd()[0]] << ' '
                      << vertices_h.view().ptv2()[vertices_h.view().sortInd()[vertices_h.view().nvFinal() - 1]]
                      << " at " << vertices_h.view().sortInd()[0] << ' '
                      << vertices_h.view().sortInd()[vertices_h.view().nvFinal() - 1] << std::endl;
          }

          float dd[vertices_h.view().nvFinal()];
          for (auto kv = 0U; kv < vertices_h.view().nvFinal(); ++kv) {
            auto zr = vertices_h.view().zv()[kv];
            auto md = 500.0f;
            for (int zint = 0; zint < ws_h.view().metadata().size(); ++zint) {
              auto d = std::abs(zr - ws_h.view().zt()[zint]);
              md = std::min(d, md);
            }
            dd[kv] = md;
          }
          if (i == 6) {
            for (auto d : dd)
              std::cout << d << ' ';
            std::cout << std::endl;
          }
          auto mx = std::minmax_element(dd, dd + vertices_h.view().nvFinal());
          float rms = 0;
          for (auto d : dd)
            rms += d * d;
          rms = std::sqrt(rms) / (vertices_h.view().nvFinal() - 1);
          std::cout << "min max rms " << *mx.first << ' ' << *mx.second << ' ' << rms << std::endl;

        }  // loop on events
      }    // lopp on ave vert
    }
  }  // namespace vertexfinder_t
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

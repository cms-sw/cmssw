#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "RecoVertex/PrimaryVertexProducer_Alpaka/plugins/alpaka/FitterAlgo.h"

//#define DEBUG_RECOVERTEX_PRIMARYVERTEXPRODUCER_ALPAKA_FITTERALGO 1

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  using namespace cms::alpakatools;

  class fitVertices {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  const portablevertex::TrackDeviceCollection::ConstView tracks,
                                  portablevertex::VertexDeviceCollection::View vertices,
                                  BeamSpotPOD const* beamSpot,
                                  bool* useBeamSpotConstraint) const {
#ifdef DEBUG_RECOVERTEX_PRIMARYVERTEXPRODUCER_ALPAKA_FITTERALGO
      if (once_per_block(acc)) {
        printf("[FitterAlgo::fitVertices()] In Vertex 0, %i tracks\n", vertices[0].ntracks());
        for (int itrackInVertex = 0; itrackInVertex < vertices[0].ntracks(); itrackInVertex++) {
          int itrack = vertices[0].track_id()[itrackInVertex];
          printf("[FitterAlgo::fitVertices()] Tracks: %i, %1.9f, %1.9f, %1.9f, %1.9f, %1.9f\n",
                 itrack,
                 tracks[itrack].x(),
                 tracks[itrack].y(),
                 tracks[itrack].z(),
                 tracks[itrack].dxy2(),
                 tracks[itrack].dz2());
        }
      }
#endif
      // These are the kernel operations themselves
      const int nTrueVertex = vertices[0].nV();  // Set max true vertex
      // Magic numbers from https://github.com/cms-sw/cmssw/blob/master/RecoVertex/PrimaryVertexProducer/interface/WeightedMeanFitter.h#L12
      const float precision = 1e-12;
      const float precisionsq = 1e-24;
      float corr_x = 1.2;
      const float corr_z = 1.4;
      const int maxIterations = 2;
      const float muSquare = 9.;
      // BeamSpot coordinates are initialized to 0, if we use beamSpot, we change them
      float bserrx = 0.;
      float bserry = 0.;
      float bsx = 0.;
      float bsy = 0.;
      if (*useBeamSpotConstraint) {
        bserrx = beamSpot->beamWidthX < precisionsq ? 1. / (precisionsq) : 1. / (beamSpot->beamWidthX);
        bserry = beamSpot->beamWidthY < precisionsq ? 1. / (precisionsq) : 1. / (beamSpot->beamWidthY);
        bsx = beamSpot->x;
        bsy = beamSpot->y;
        corr_x = 1.0;
      }
#ifdef DEBUG_RECOVERTEX_PRIMARYVERTEXPRODUCER_ALPAKA_FITTERALGO
      printf("[FitterAlgo::fitVertices()] Set-up, beamspot constrains: %1.9f, %1.9f, %1.9f, %1.9f\n",
             bserrx,
             bserry,
             bsx,
             bsy);
#endif
      for (auto i : uniform_elements(
               acc,
               nTrueVertex)) {  // By construction nTrueVertex <= 512, so this will always be a 1 thread to 1 vertex assignment
        if (not(vertices[i].isGood()))
          continue;  // If vertex was killed before, just skip
                     // Initialize positions and errors to 0
#ifdef DEBUG_RECOVERTEX_PRIMARYVERTEXPRODUCER_ALPAKA_FITTERALGO
        printf("[FitterAlgo::fitVertices()] Start vertex %i with %i tracks\n", i, vertices[i].ntracks());
#endif
        float x = 0.;
        float y = 0.;
        float z = 0.;
        float errx = 0.;
        float errz = 0.;

        for (int itrackInVertex = 0; itrackInVertex < vertices[i].ntracks(); itrackInVertex++) {
          int itrack = vertices[i].track_id()[itrackInVertex];
          float wxy = tracks[itrack].dxy2() <= precisionsq ? 1. / precisionsq : 1. / tracks[itrack].dxy2();
          float wz = tracks[itrack].dz2() <= precisionsq ? 1. / precisionsq : 1. / tracks[itrack].dz2();
          x += tracks[itrack].x() * wxy;
          y += tracks[itrack].y() * wxy;
          z += tracks[itrack].z() * wz;
          errx += wxy;  // x and y have the same error due to symmetry
          errz += wz;
        }
        float erry = errx;
#ifdef DEBUG_RECOVERTEX_PRIMARYVERTEXPRODUCER_ALPAKA_FITTERALGO
        printf("[FitterAlgo::fitVertices()] After first iteration, before dividing, %1.9f %1.9f %1.9f %1.9f %1.9f \n",
               x,
               y,
               z,
               errx,
               errz);
#endif
        // Now add the BeamSpot and get first estimation, if no beamspot, this changes nothing
        x = (x + bsx * bserrx * bserrx) / (bserrx * bserrx + errx);
        y = (y + bsy * bserry * bserry) / (bserry * bserry + erry);
        z /= errz;
#ifdef DEBUG_RECOVERTEX_PRIMARYVERTEXPRODUCER_ALPAKA_FITTERALGO
        printf("[FitterAlgo::fitVertices()] After first iteration, after dividing, %1.9f %1.9f %1.9f %1.9f %1.9f \n",
               x,
               y,
               z,
               errx,
               errz);
#endif
        // Weights and square weights for iteration
        float s_wx, s_wz;
        errx = 1 / errx;
        erry = 1 / erry;
        errz = 1 / errz;
        int ndof;
        // Run iterative weighted mean fitter
        int niter = 0;
        float old_x;
        float old_y;
        float old_z;
        while ((niter++) < maxIterations) {
#ifdef DEBUG_RECOVERTEX_PRIMARYVERTEXPRODUCER_ALPAKA_FITTERALGO
          printf(
              "[FitterAlgo::fitVertices()] At iteration %i, errs are %1.15f %1.15f %1.15f\n", niter, errx, erry, errz);
#endif
          old_x = x;
          old_y = y;
          old_z = z;
          s_wx = 0.;
          s_wz = 0.;
          x = 0.;
          y = 0.;
          z = 0.;
          ndof = 0;
          for (int itrackInVertex = 0; itrackInVertex < vertices[i].ntracks(); itrackInVertex++) {
            int itrack = vertices[i].track_id()[itrackInVertex];
            // Position (ref point) of the track
            double tx = tracks[itrack].x();
            double ty = tracks[itrack].y();
            double tz = tracks[itrack].z();
            // Momentum of the track
            double px = tracks[itrack].px();
            double py = tracks[itrack].py();
            double pz = tracks[itrack].pz();
            // To compute the PCA of the track to the current vertex
            double pnorm2 = px * px + py * py + pz * pz;
            // This is the 'time' needed to move from the ref point to the PCA scalar product of (x_v-x_t)*p_t over magnitude squared of p_t
            double t = (px * (old_x - tx) + py * (old_y - ty) + pz * (old_z - tz)) / pnorm2;
#ifdef DEBUG_RECOVERTEX_PRIMARYVERTEXPRODUCER_ALPAKA_FITTERALGO
            printf(
                "[FitterAlgo::fitVertices()] Track x: %1.9f, y: %1.9f, z:%1.9f, px: %1.9f, py: %1.9f, pz: %1.9f, "
                "t:%1.9f\n",
                tx,
                ty,
                tz,
                px,
                py,
                pz,
                t);
#endif
            // Advance the track until the PCA
            tx += px * t;
            ty += py * t;
            tz += pz * t;
            float wx = tracks[itrack].dxy2() <= precisionsq ? 1. / precisionsq : 1. / tracks[itrack].dxy2();
            float wz = tracks[itrack].dz2() <= precisionsq ? 1. / precisionsq : 1. / tracks[itrack].dz2();
#ifdef DEBUG_RECOVERTEX_PRIMARYVERTEXPRODUCER_ALPAKA_FITTERALGO
            printf("[FitterAlgo::fitVertices()] Track wx: %1.9f, wz: %1.9f\n", wx, wz);
            printf("[FitterAlgo::fitVertices()] Track sigmas: %1.3f %1.3f %1.3f\n",
                   (tx - old_x) * (tx - old_x) / (1 / wx + errx),
                   (ty - old_y) * (ty - old_y) / (1 / wx + erry),
                   (tz - old_z) * (tz - old_z) / (1 / wz + errz));
            printf("[FitterAlgo::fitVertices()] Track bools: %i %i %i\n",
                   ((tx - old_x) * (tx - old_x) / (1 / wx + errx) < muSquare),
                   ((ty - old_y) * (ty - old_y) / (1 / wx + erry) < muSquare),
                   ((tz - old_z) * (tz - old_z) / (1 / wz + errz) < muSquare));
#endif
            if (((tx - old_x) * (tx - old_x) / (1 / wx + errx) < muSquare) &&
                ((ty - old_y) * (ty - old_y) / (1 / wx + erry) < muSquare) &&
                ((tz - old_z) * (tz - old_z) / (1 / wz + errz) <
                 muSquare)) {  // I.e., old coordinates of PCA are within 3 sigma of current vertex position, keep the track
              ndof += 1;
              vertices[i].track_weight()[itrackInVertex] = 1;
              s_wx += wx;
              s_wz += wz;
            } else {  // Otherwise, discard track
              vertices[i].track_weight()[itrackInVertex] = 0;
              wx = 0.;
              wz = 0.;
            }
#ifdef DEBUG_RECOVERTEX_PRIMARYVERTEXPRODUCER_ALPAKA_FITTERALGO
            printf("[FitterAlgo::fitVertices()] Track %i weights after %1.10f, %1.10f\n", itrackInVertex, wx, wz);
#endif
            // Here, will only change if track is within 3 sigma
            x += tx * wx;
            y += ty * wx;
            z += tz * wz;
#ifdef DEBUG_RECOVERTEX_PRIMARYVERTEXPRODUCER_ALPAKA_FITTERALGO
            printf("[FitterAlgo::fitVertices()] Track adds x: %1.9f, y: %1.9f z: %1.9f\n", tx * wx, ty * wx, tz * wz);
#endif
          }  // end for
// After all tracks, add BS uncertainties, will do nothing if not used
#ifdef DEBUG_RECOVERTEX_PRIMARYVERTEXPRODUCER_ALPAKA_FITTERALGO
          printf("[FitterAlgo::fitVertices()] Before adding BS in %i iteration %1.9f %1.9f %1.9f %1.9f %1.9f %1.9f \n",
                 niter,
                 x,
                 y,
                 z,
                 s_wx,
                 s_wx,
                 s_wz);
#endif
          x += bsx * bserrx;
          y += bsy * bserry;
#ifdef DEBUG_RECOVERTEX_PRIMARYVERTEXPRODUCER_ALPAKA_FITTERALGO
          printf("[FitterAlgo::fitVertices()] BS adds x: %1.9f, y: %1.9f\n", bsx * bserrx, bsy * bserry);
#endif
          float s_wy = s_wx;
          s_wx += bserrx;
          s_wy += bserry;
#ifdef DEBUG_RECOVERTEX_PRIMARYVERTEXPRODUCER_ALPAKA_FITTERALGO
          printf("[FitterAlgo::fitVertices()] Before dividing %i iteration %1.9f %1.9f %1.9f %1.9f %1.9f %1.9f \n",
                 niter,
                 x,
                 y,
                 z,
                 s_wx,
                 s_wy,
                 s_wz);
#endif
          x /= s_wx;
          y /= s_wy;
          z /= s_wz;
          errx = 1 / s_wx;
          errz = 1 / s_wz;
          erry = 1 / s_wy;
#ifdef DEBUG_RECOVERTEX_PRIMARYVERTEXPRODUCER_ALPAKA_FITTERALGO
          printf("[FitterAlgo::fitVertices()] After dividing %i iteration %1.9f %1.9f %1.9f %1.9f %1.9f \n",
                 niter,
                 x,
                 y,
                 z,
                 errx,
                 errz);
          printf("[FitterAlgo::fitVertices()] Compare old and new: %1.9f %1.9f, %1.9f %1.9f, %1.9f %1.9f \n",
                 old_x,
                 x,
                 old_y,
                 y,
                 old_z,
                 z);
#endif
          if ((abs(old_x - x) < precision) && (abs(old_y - y) < precision) && (abs(old_z - z) < precision))
            break;  // If good enough, stop the iterations
        }  // end while
        // Assign everything back in global memory to get the fitted vertex!
        errx *= corr_x * corr_x;
        erry *= corr_x * corr_x;
        errz *= corr_z * corr_z;
        vertices[i].x() = x;
        vertices[i].y() = y;
        vertices[i].z() = z;
        vertices[i].errx() = errx;
        vertices[i].erry() = erry;
        vertices[i].errz() = errz;
        vertices[i].ndof() = ndof;
        // Last get the chi square of the final vertex fit
        double chi2 = 0.;
        for (int itrackInVertex = 0; itrackInVertex < vertices[i].ntracks(); itrackInVertex++) {
          int itrack = vertices[i].track_id()[itrackInVertex];
          // Position (ref point) of the track
          float tx = tracks[itrack].x();
          float ty = tracks[itrack].y();
          float tz = tracks[itrack].z();
          float wx = tracks[itrack].dxy2();
          float wz = tracks[itrack].dz2();
          chi2 +=
              (tx - x) * (tx - x) / (errx + wx) + (ty - y) * (ty - y) / (erry + wx) +
              (tz - z) * (tz - z) /
                  (errz +
                   wz);  // chi2 doesn't use the PCA distance, but the ref point coordinates as in https://github.com/cms-sw/cmssw/blob/master/RecoVertex/PrimaryVertexProducer/interface/WeightedMeanFitter.h#L316
        }  // end for
        vertices[i].chi2() = chi2;
#ifdef DEBUG_RECOVERTEX_PRIMARYVERTEXPRODUCER_ALPAKA_FITTERALGO
        printf(
            "[FitterAlgo::fitVertices()] Vertex %i, x: %1.9f, y:%1.9f, z:%1.9f, errx:%1.9f, errz:%1.9f, chi2:%1.9f, "
            "ndof:%1.9f\n",
            i,
            vertices[i].x(),
            vertices[i].y(),
            vertices[i].z(),
            vertices[i].errx(),
            vertices[i].errz(),
            vertices[i].chi2(),
            vertices[i].ndof());
#endif
      }  // end for (stride) loop
    }  // operator()
  };  // class fitVertices

  FitterAlgo::FitterAlgo(Queue& queue, const int32_t nV, fitterParameters fPar)
      : useBeamSpotConstraint(cms::alpakatools::make_device_buffer<bool>(queue)) {
    // Set fitter parameters
    alpaka::memset(queue, useBeamSpotConstraint, fPar.useBeamSpotConstraint);
  }  // FitterAlgo::FitterAlgo

  void FitterAlgo::fit(Queue& queue,
                       const portablevertex::TrackDeviceCollection& deviceTrack,
                       portablevertex::VertexDeviceCollection& deviceVertex,
                       const BeamSpotDevice& deviceBeamSpot) {
    const int nVertexToFit =
        512;  // Right now it executes for all 512 vertex, even if vertex collection is empty (in which case the kernel passes). Can we make this dynamic to vertex size?
    const int threadsPerBlock = 32;
    const int blocks = divide_up_by(nVertexToFit, threadsPerBlock);
    alpaka::exec<Acc1D>(queue,
                        make_workdiv<Acc1D>(blocks, threadsPerBlock),
                        fitVertices{},
                        deviceTrack.view(),
                        deviceVertex.view(),
                        deviceBeamSpot.data(),
                        useBeamSpotConstraint.data());
  }  // FitterAlgo::fit
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

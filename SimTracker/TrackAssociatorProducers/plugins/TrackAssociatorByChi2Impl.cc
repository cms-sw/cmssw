#include "TrackAssociatorByChi2Impl.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"

#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/GeometrySurface/interface/Line.h"
#include "SimTracker/TrackAssociation/interface/trackAssociationChi2.h"
#include "TrackingTools/PatternTools/interface/trackingParametersAtClosestApproachToBeamSpot.h"

using namespace edm;
using namespace reco;
using namespace std;
using namespace track_associator;

RecoToSimCollection TrackAssociatorByChi2Impl::associateRecoToSim(
    const RefToBaseVector<Track>& tC, const RefVector<TrackingParticleCollection>& tPCH) const {
  const BeamSpot& bs = *beamSpot_;

  RecoToSimCollection outputCollection(productGetter_);

  //dereference the Refs only once and precompute params
  std::vector<TrackingParticle const*> tPC;
  std::vector<std::pair<bool, TrackBase::ParameterVector>> tpParams;
  tPC.reserve(tPCH.size());
  tpParams.reserve(tPCH.size());
  for (auto const& ref : tPCH) {
    auto const& tp = *ref;
    tPC.push_back(&tp);

    int charge = tp.charge();
    if (charge == 0)
      tpParams.emplace_back(false, TrackBase::ParameterVector());
    else {
      using BVec = Basic3DVector<double>;
      tpParams.emplace_back(
          trackingParametersAtClosestApproachToBeamSpot(BVec(tp.vertex()), BVec(tp.momentum()), charge, *mF_, bs));
    }
  }

  int tindex = 0;
  for (RefToBaseVector<Track>::const_iterator rt = tC.begin(); rt != tC.end(); rt++, tindex++) {
    LogDebug("TrackAssociator") << "=========LOOKING FOR ASSOCIATION==========="
                                << "\n"
                                << "rec::Track #" << tindex << " with pt=" << (*rt)->pt() << "\n"
                                << "==========================================="
                                << "\n";

    TrackBase::ParameterVector rParameters = (*rt)->parameters();

    TrackBase::CovarianceMatrix recoTrackCovMatrix = (*rt)->covariance();
    if (onlyDiagonal_) {
      for (unsigned int i = 0; i < 5; i++) {
        for (unsigned int j = 0; j < 5; j++) {
          if (i != j)
            recoTrackCovMatrix(i, j) = 0;
        }
      }
    }

    recoTrackCovMatrix.Invert();

    int tpindex = 0;
    for (auto tp = tPC.begin(); tp != tPC.end(); tp++, ++tpindex) {
      //skip tps with a very small pt
      //if (sqrt((*tp)->momentum().perp2())<0.5) continue;
      if (!tpParams[tpindex].first)
        continue;

      double chi2 = trackAssociationChi2(rParameters, recoTrackCovMatrix, tpParams[tpindex].second);

      if (chi2 < chi2cut_) {
        //-chi2 because the Association Map is ordered using std::greater
        outputCollection.insert(tC[tindex], std::make_pair(tPCH[tpindex], -chi2));
      }
    }
  }
  outputCollection.post_insert();
  return outputCollection;
}

SimToRecoCollection TrackAssociatorByChi2Impl::associateSimToReco(
    const RefToBaseVector<Track>& tC, const RefVector<TrackingParticleCollection>& tPCH) const {
  const BeamSpot& bs = *beamSpot_;

  SimToRecoCollection outputCollection(productGetter_);

  //compute track parameters only once
  std::vector<TrackBase::ParameterVector> tPars;
  tPars.reserve(tC.size());
  std::vector<TrackBase::CovarianceMatrix> tCovs;
  tCovs.reserve(tC.size());
  for (auto const& ref : tC) {
    auto const& aTk = *ref;
    tPars.emplace_back(aTk.parameters());

    TrackBase::CovarianceMatrix recoTrackCovMatrix = aTk.covariance();
    if (onlyDiagonal_) {
      for (unsigned int i = 0; i < 5; i++) {
        for (unsigned int j = 0; j < 5; j++) {
          if (i != j)
            recoTrackCovMatrix(i, j) = 0;
        }
      }
    }
    recoTrackCovMatrix.Invert();
    tCovs.emplace_back(recoTrackCovMatrix);
  }

  int tpindex = 0;
  for (auto tp = tPCH.begin(); tp != tPCH.end(); tp++, ++tpindex) {
    //skip tps with a very small pt
    //if (sqrt(tp->momentum().perp2())<0.5) continue;
    auto const& aTP = **tp;
    int charge = aTP.charge();
    if (charge == 0)
      continue;

    LogDebug("TrackAssociator") << "=========LOOKING FOR ASSOCIATION==========="
                                << "\n"
                                << "TrackingParticle #" << tpindex << " with pt=" << sqrt(aTP.momentum().perp2())
                                << "\n"
                                << "==========================================="
                                << "\n";

    using BVec = Basic3DVector<double>;
    auto const tpBoolParams =
        trackingParametersAtClosestApproachToBeamSpot(BVec(aTP.vertex()), BVec(aTP.momentum()), charge, *mF_, bs);
    if (!tpBoolParams.first)
      continue;

    for (unsigned int tindex = 0; tindex < tC.size(); tindex++) {
      TrackBase::ParameterVector const& rParameters = tPars[tindex];
      TrackBase::CovarianceMatrix const& recoTrackCovMatrix = tCovs[tindex];

      double chi2 = trackAssociationChi2(rParameters, recoTrackCovMatrix, tpBoolParams.second);

      if (chi2 < chi2cut_) {
        //-chi2 because the Association Map is ordered using std::greater
        outputCollection.insert(*tp, std::make_pair(tC[tindex], -chi2));
      }
    }
  }
  outputCollection.post_insert();
  return outputCollection;
}

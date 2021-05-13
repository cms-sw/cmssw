#include "TrackAssociatorByChi2Impl.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"

#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/GeometrySurface/interface/Line.h"
#include "SimTracker/TrackAssociation/interface/trackAssociationChi2.h"

using namespace edm;
using namespace reco;
using namespace std;

double TrackAssociatorByChi2Impl::getChi2(const TrackBase::ParameterVector& rParameters,
                                          const TrackBase::CovarianceMatrix& recoTrackCovMatrix,
                                          const Basic3DVector<double>& momAtVtx,
                                          const Basic3DVector<double>& vert,
                                          int charge,
                                          const reco::BeamSpot& bs) const {
  return track_associator::trackAssociationChi2(rParameters, recoTrackCovMatrix, momAtVtx, vert, charge, *mF_, bs);
}

RecoToSimCollection TrackAssociatorByChi2Impl::associateRecoToSim(
    const edm::RefToBaseVector<reco::Track>& tC, const edm::RefVector<TrackingParticleCollection>& tPCH) const {
  const reco::BeamSpot& bs = *beamSpot_;

  RecoToSimCollection outputCollection(productGetter_);

  //dereference the edm::Refs only once
  std::vector<TrackingParticle const*> tPC;
  tPC.reserve(tPCH.size());
  for (auto const& ref : tPCH) {
    tPC.push_back(&(*ref));
  }

  int tindex = 0;
  for (RefToBaseVector<reco::Track>::const_iterator rt = tC.begin(); rt != tC.end(); rt++, tindex++) {
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
      int charge = (*tp)->charge();
      if (charge == 0)
        continue;
      Basic3DVector<double> momAtVtx((*tp)->momentum());
      Basic3DVector<double> vert((*tp)->vertex());

      double chi2 = getChi2(rParameters, recoTrackCovMatrix, momAtVtx, vert, charge, bs);

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
    const edm::RefToBaseVector<reco::Track>& tC, const edm::RefVector<TrackingParticleCollection>& tPCH) const {
  const reco::BeamSpot& bs = *beamSpot_;

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

    Basic3DVector<double> momAtVtx(aTP.momentum());
    Basic3DVector<double> vert(aTP.vertex());

    for (unsigned int tindex = 0; tindex < tC.size(); tindex++) {
      TrackBase::ParameterVector const& rParameters = tPars[tindex];
      TrackBase::CovarianceMatrix const& recoTrackCovMatrix = tCovs[tindex];

      double chi2 = getChi2(rParameters, recoTrackCovMatrix, momAtVtx, vert, charge, bs);

      if (chi2 < chi2cut_) {
        //-chi2 because the Association Map is ordered using std::greater
        outputCollection.insert(*tp, std::make_pair(tC[tindex], -chi2));
      }
    }
  }
  outputCollection.post_insert();
  return outputCollection;
}

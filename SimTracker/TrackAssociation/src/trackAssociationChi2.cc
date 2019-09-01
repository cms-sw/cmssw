#include "DataFormats/Math/interface/deltaPhi.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimTracker/TrackAssociation/interface/trackAssociationChi2.h"
#include "TrackingTools/PatternTools/interface/trackingParametersAtClosestApproachToBeamSpot.h"

namespace track_associator {
  constexpr double invalidChi2 = 10000000000.;

  double trackAssociationChi2(const reco::TrackBase::ParameterVector &rParameters,
                              const reco::TrackBase::CovarianceMatrix &recoTrackCovMatrix,
                              const Basic3DVector<double> &momAtVtx,
                              const Basic3DVector<double> &vert,
                              int charge,
                              const MagneticField &magfield,
                              const reco::BeamSpot &bs) {
    double chi2 = invalidChi2;

    std::pair<bool, reco::TrackBase::ParameterVector> params =
        reco::trackingParametersAtClosestApproachToBeamSpot(vert, momAtVtx, charge, magfield, bs);
    if (params.first) {
      reco::TrackBase::ParameterVector sParameters = params.second;

      reco::TrackBase::ParameterVector diffParameters = rParameters - sParameters;
      diffParameters[2] = reco::deltaPhi(diffParameters[2], 0.f);
      chi2 = ROOT::Math::Dot(diffParameters * recoTrackCovMatrix, diffParameters);
      chi2 /= 5;

      LogDebug("TrackAssociator") << "====NEW RECO TRACK WITH PT="
                                  << sin(rParameters[1]) * float(charge) / rParameters[0] << "====\n"
                                  << "qoverp sim: " << sParameters[0] << "\n"
                                  << "lambda sim: " << sParameters[1] << "\n"
                                  << "phi    sim: " << sParameters[2] << "\n"
                                  << "dxy    sim: " << sParameters[3] << "\n"
                                  << "dsz    sim: " << sParameters[4] << "\n"
                                  << ": " /*<< */ << "\n"
                                  << "qoverp rec: " << rParameters[0] << "\n"
                                  << "lambda rec: " << rParameters[1] << "\n"
                                  << "phi    rec: " << rParameters[2] << "\n"
                                  << "dxy    rec: " << rParameters[3] << "\n"
                                  << "dsz    rec: " << rParameters[4] << "\n"
                                  << ": " /*<< */ << "\n"
                                  << "chi2: " << chi2 << "\n";
    }
    return chi2;
  }

  double trackAssociationChi2(const reco::TrackBase::ParameterVector &rParameters,
                              const reco::TrackBase::CovarianceMatrix &recoTrackCovMatrix,
                              const TrackingParticle &trackingParticle,
                              const MagneticField &magfield,
                              const reco::BeamSpot &bs) {
    const int charge = trackingParticle.charge();
    if (charge == 0)
      return invalidChi2;

    const auto tpMom = trackingParticle.momentum();
    Basic3DVector<double> momAtVtx(tpMom.x(), tpMom.y(), tpMom.z());
    Basic3DVector<double> vert(trackingParticle.vertex());

    return trackAssociationChi2(rParameters, recoTrackCovMatrix, momAtVtx, vert, charge, magfield, bs);
  }

  double trackAssociationChi2(const reco::TrackBase &track,
                              const TrackingParticle &trackingParticle,
                              const MagneticField &magfield,
                              const reco::BeamSpot &bs) {
    auto cov = track.covariance();
    cov.Invert();

    return trackAssociationChi2(track.parameters(), cov, trackingParticle, magfield, bs);
  }
}  // namespace track_associator

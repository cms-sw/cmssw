#include "RecoTracker/PixelVertexFinding/interface/PVClusterComparer.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "TMath.h"

PVClusterComparer::PVClusterComparer()
    : track_pT_min_(2.5), track_pT_max_(10.), track_chi2_max_(9999999.), track_prob_min_(-1.) {
  setChisquareQuantile();
}

PVClusterComparer::PVClusterComparer(double track_pt_min,
                                     double track_pt_max,
                                     double track_chi2_max,
                                     double track_prob_min)
    : track_pT_min_(track_pt_min),
      track_pT_max_(track_pt_max),
      track_chi2_max_(track_chi2_max),
      track_prob_min_(track_prob_min) {
  setChisquareQuantile();
}

double PVClusterComparer::pTSquaredSum(const PVCluster &v) {
  double sum = 0;
  for (size_t i = 0; i < v.tracks().size(); ++i) {
    double pt = v.tracks()[i]->pt();
    if (pt < track_pT_min_)
      continue;  // Don't count tracks below track_pT_min_ (2.5 GeV)

    // RM : exclude badly reconstructed tracks from the sum
    //    if (track_prob_min_ >= 0. && track_prob_min_ <= 1.)
    //      if (TMath::Prob(v.tracks()[i]->chi2(),v.tracks()[i]->ndof()) < track_prob_min_) continue ;
    if (track_prob_min_ >= 0. && track_prob_min_ <= 1.) {
      size_t ndof = v.tracks()[i]->ndof();
      if (ndof >= maxChi2_.size())
        updateChisquareQuantile(ndof);
      // cut on chi2 which corresponds to the configured probability
      if (v.tracks()[i]->chi2() > maxChi2_[ndof])
        continue;
    }
    if (v.tracks()[i]->normalizedChi2() > track_chi2_max_)
      continue;
    if (pt > track_pT_max_)
      pt = track_pT_max_;
    sum += pt * pt;
  }
  return sum;
}

double PVClusterComparer::pTSquaredSum(const reco::Vertex &v) {
  double sum = 0;
  for (reco::Vertex::trackRef_iterator i = v.tracks_begin(), ie = v.tracks_end(); i != ie; ++i) {
    double pt = (*i)->pt();
    if (pt < track_pT_min_)
      continue;  // Don't count tracks below track_pT_min_ (2.5 GeV)

    // RM : exclude badly reconstructed tracks from the sum
    //    if (track_prob_min_ >= 0. && track_prob_min_ <= 1.)
    //      if (TMath::Prob((*i)->chi2(),(*i)->ndof()) < track_prob_min_) continue ;
    if (track_prob_min_ >= 0. && track_prob_min_ <= 1.) {
      unsigned int ndof = (*i)->ndof();
      if (ndof >= maxChi2_.size())
        updateChisquareQuantile(ndof);
      // cut on chi2 which corresponds to the configured probability
      if ((*i)->chi2() > maxChi2_[ndof])
        continue;
    }
    if ((*i)->normalizedChi2() > track_chi2_max_)
      continue;

    if (pt > track_pT_max_)
      pt = track_pT_max_;
    sum += pt * pt;
  }
  return sum;
}

void PVClusterComparer::setChisquareQuantile() {
  maxChi2_.clear();
  maxChi2_.resize(20, 0.0);
  if (track_prob_min_ >= 0. && track_prob_min_ <= 1.)
    for (size_t ndof = 0; ndof < maxChi2_.size(); ++ndof)
      // http://root.cern.ch/root/html/TMath.html#TMath:ChisquareQuantile
      maxChi2_[ndof] = TMath::ChisquareQuantile(1 - track_prob_min_, ndof);
}

void PVClusterComparer::updateChisquareQuantile(size_t ndof) {
  size_t oldsize = maxChi2_.size();
  //	maxChi2_.resize(ndof+1);
  for (size_t i = oldsize; i <= ndof; ++i) {
    double chi2 = TMath::ChisquareQuantile(1 - track_prob_min_, i);
    maxChi2_.push_back(chi2);
  }
}

bool PVClusterComparer::operator()(const PVCluster &v1, const PVCluster &v2) {
  return (pTSquaredSum(v1) > pTSquaredSum(v2));
}
bool PVClusterComparer::operator()(const reco::Vertex &v1, const reco::Vertex &v2) {
  return (pTSquaredSum(v1) > pTSquaredSum(v2));
}

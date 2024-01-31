#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ValidatedPluginMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "vdt/vdtMath.h"

#include "RecoVertex/PrimaryVertexProducer/interface/VertexTimeAlgorithmFromTracksPID.h"

#ifdef PVTX_DEBUG
#define LOG edm::LogPrint("VertexTimeAlgorithmFromTracksPID")
#else
#define LOG LogDebug("VertexTimeAlgorithmFromTracksPID")
#endif

VertexTimeAlgorithmFromTracksPID::VertexTimeAlgorithmFromTracksPID(edm::ParameterSet const& iConfig,
                                                                   edm::ConsumesCollector& iCC)
    : VertexTimeAlgorithmBase(iConfig, iCC),
      trackMTDTimeToken_(iCC.consumes(iConfig.getParameter<edm::InputTag>("trackMTDTimeVMapTag"))),
      trackMTDTimeErrorToken_(iCC.consumes(iConfig.getParameter<edm::InputTag>("trackMTDTimeErrorVMapTag"))),
      trackMTDTimeQualityToken_(iCC.consumes(iConfig.getParameter<edm::InputTag>("trackMTDTimeQualityVMapTag"))),
      trackMTDTofPiToken_(iCC.consumes(iConfig.getParameter<edm::InputTag>("trackMTDTofPiVMapTag"))),
      trackMTDTofKToken_(iCC.consumes(iConfig.getParameter<edm::InputTag>("trackMTDTofKVMapTag"))),
      trackMTDTofPToken_(iCC.consumes(iConfig.getParameter<edm::InputTag>("trackMTDTofPVMapTag"))),
      minTrackVtxWeight_(iConfig.getParameter<double>("minTrackVtxWeight")),
      minTrackTimeQuality_(iConfig.getParameter<double>("minTrackTimeQuality")),
      probPion_(iConfig.getParameter<double>("probPion")),
      probKaon_(iConfig.getParameter<double>("probKaon")),
      probProton_(iConfig.getParameter<double>("probProton")),
      Tstart_(iConfig.getParameter<double>("Tstart")),
      coolingFactor_(iConfig.getParameter<double>("coolingFactor")) {}

void VertexTimeAlgorithmFromTracksPID::fillPSetDescription(edm::ParameterSetDescription& iDesc) {
  VertexTimeAlgorithmBase::fillPSetDescription(iDesc);

  iDesc.add<edm::InputTag>("trackMTDTimeVMapTag", edm::InputTag("trackExtenderWithMTD:generalTracktmtd"))
      ->setComment("Input ValueMap for track time at MTD");
  iDesc.add<edm::InputTag>("trackMTDTimeErrorVMapTag", edm::InputTag("trackExtenderWithMTD:generalTracksigmatmtd"))
      ->setComment("Input ValueMap for track time uncertainty at MTD");
  iDesc.add<edm::InputTag>("trackMTDTimeQualityVMapTag", edm::InputTag("mtdTrackQualityMVA:mtdQualMVA"))
      ->setComment("Input ValueMap for track MVA quality value");
  iDesc.add<edm::InputTag>("trackMTDTofPiVMapTag", edm::InputTag("trackExtenderWithMTD:generalTrackTofPi"))
      ->setComment("Input ValueMap for track tof as pion");
  iDesc.add<edm::InputTag>("trackMTDTofKVMapTag", edm::InputTag("trackExtenderWithMTD:generalTrackTofK"))
      ->setComment("Input ValueMap for track tof as kaon");
  iDesc.add<edm::InputTag>("trackMTDTofPVMapTag", edm::InputTag("trackExtenderWithMTD:generalTrackTofP"))
      ->setComment("Input ValueMap for track tof as proton");

  iDesc.add<double>("minTrackVtxWeight", 0.5)->setComment("Minimum track weight");
  iDesc.add<double>("minTrackTimeQuality", 0.8)->setComment("Minimum MVA Quality selection on tracks");

  iDesc.add<double>("probPion", 0.7)->setComment("A priori probability pions");
  iDesc.add<double>("probKaon", 0.2)->setComment("A priori probability kaons");
  iDesc.add<double>("probProton", 0.1)->setComment("A priori probability protons");

  iDesc.add<double>("Tstart", 256.)->setComment("DA initial temperature T");
  iDesc.add<double>("coolingFactor", 0.5)->setComment("DA cooling factor");
}

void VertexTimeAlgorithmFromTracksPID::setEvent(edm::Event& iEvent, edm::EventSetup const&) {
  // additional collections required for vertex-time calculation
  trackMTDTimes_ = iEvent.get(trackMTDTimeToken_);
  trackMTDTimeErrors_ = iEvent.get(trackMTDTimeErrorToken_);
  trackMTDTimeQualities_ = iEvent.get(trackMTDTimeQualityToken_);
  trackMTDTofPi_ = iEvent.get(trackMTDTofPiToken_);
  trackMTDTofK_ = iEvent.get(trackMTDTofKToken_);
  trackMTDTofP_ = iEvent.get(trackMTDTofPToken_);
}

bool VertexTimeAlgorithmFromTracksPID::vertexTime(float& vtxTime,
                                                  float& vtxTimeError,
                                                  const TransientVertex& vtx) const {
  if (vtx.originalTracks().empty()) {
    return false;
  }

  auto const vtxTime_init = vtxTime;
  auto const vtxTimeError_init = vtxTimeError;
  const int max_iterations = 100;

  double tsum = 0;
  double wsum = 0;
  double w2sum = 0;

  double const a[3] = {probPion_, probKaon_, probProton_};

  std::vector<TrackInfo> v_trackInfo;
  v_trackInfo.reserve(vtx.originalTracks().size());

  // initial guess
  for (const auto& trk : vtx.originalTracks()) {
    auto const trkWeight = vtx.trackWeight(trk);
    if (trkWeight > minTrackVtxWeight_) {
      auto const trkTimeQuality = trackMTDTimeQualities_[trk.trackBaseRef()];

      if (trkTimeQuality >= minTrackTimeQuality_) {
        auto const trkTime = trackMTDTimes_[trk.trackBaseRef()];
        auto const trkTimeError = trackMTDTimeErrors_[trk.trackBaseRef()];

        v_trackInfo.emplace_back();
        auto& trkInfo = v_trackInfo.back();

        trkInfo.trkWeight = trkWeight;
        trkInfo.trkTimeError = trkTimeError;

        trkInfo.trkTimeHyp[0] = trkTime - trackMTDTofPi_[trk.trackBaseRef()];
        trkInfo.trkTimeHyp[1] = trkTime - trackMTDTofK_[trk.trackBaseRef()];
        trkInfo.trkTimeHyp[2] = trkTime - trackMTDTofP_[trk.trackBaseRef()];

        auto const wgt = trkWeight / (trkTimeError * trkTimeError);
        wsum += wgt;

        for (uint j = 0; j < 3; ++j) {
          tsum += wgt * trkInfo.trkTimeHyp[j] * a[j];
        }
        LOG << "vertexTimeFromTracks:     track"
            << " pt=" << trk.track().pt() << " eta=" << trk.track().eta() << " phi=" << trk.track().phi()
            << " vtxWeight=" << trkWeight << " time=" << trkTime << " timeError=" << trkTimeError
            << " timeQuality=" << trkTimeQuality << " timeHyp[pion]=" << trkInfo.trkTimeHyp[0]
            << " timeHyp[kaon]=" << trkInfo.trkTimeHyp[1] << " timeHyp[proton]=" << trkInfo.trkTimeHyp[2];
      }
    }
  }
  if (wsum > 0) {
    auto t0 = tsum / wsum;
    auto beta = 1. / Tstart_;
    int nit = 0;
    while ((nit++) < max_iterations) {
      tsum = 0;
      wsum = 0;
      w2sum = 0;

      for (auto const& trkInfo : v_trackInfo) {
        double dt = trkInfo.trkTimeError;
        double e[3] = {0, 0, 0};
        const double cut_off = 4.5;
        double Z = vdt::fast_exp(
            -beta * cut_off);  // outlier rejection term Z_0 = exp(-beta * cut_off) = exp(-beta * 0.5 * 3 * 3)
        for (unsigned int j = 0; j < 3; j++) {
          auto const tpull = (trkInfo.trkTimeHyp[j] - t0) / dt;
          e[j] = vdt::fast_exp(-0.5 * beta * tpull * tpull);
          Z += a[j] * e[j];
        }

        double wsum_trk = 0;
        for (uint j = 0; j < 3; j++) {
          double wt = a[j] * e[j] / Z;
          double w = wt * trkInfo.trkWeight / (dt * dt);
          wsum_trk += w;
          tsum += w * trkInfo.trkTimeHyp[j];
        }

        wsum += wsum_trk;
        w2sum += wsum_trk * wsum_trk * (dt * dt) / trkInfo.trkWeight;
      }

      if (wsum < 1e-10) {
        LOG << "vertexTimeFromTracks:   failed while iterating";
        return false;
      }

      vtxTime = tsum / wsum;

      LOG << "vertexTimeFromTracks:   iteration=" << nit << ", T= " << 1 / beta << ", t=" << vtxTime
          << ", t-t0=" << vtxTime - t0;

      if ((std::abs(vtxTime - t0) < 1e-4 / std::sqrt(beta)) and beta >= 1.) {
        vtxTimeError = std::sqrt(w2sum) / wsum;

        LOG << "vertexTimeFromTracks:   tfit = " << vtxTime << " +/- " << vtxTimeError << " trec = " << vtx.time()
            << ", iteration=" << nit;

        return true;
      }

      if ((std::abs(vtxTime - t0) < 1e-3) and beta < 1.) {
        beta = std::min(1., beta / coolingFactor_);
      }

      t0 = vtxTime;
    }

    LOG << "vertexTimeFromTracks: failed to converge";
  } else {
    LOG << "vertexTimeFromTracks: has no track timing info";
  }

  vtxTime = vtxTime_init;
  vtxTimeError = vtxTimeError_init;

  return false;
}

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "RecoVertex/PrimaryVertexProducer/interface/VertexTimeAlgorithmLegacy4D.h"

#ifdef PVTX_DEBUG
#define LOG edm::LogPrint("VertexTimeAlgorithmLegacy4D")
#else
#define LOG LogDebug("VertexTimeAlgorithmLegacy4D")
#endif

VertexTimeAlgorithmLegacy4D::VertexTimeAlgorithmLegacy4D(edm::ParameterSet const& iConfig, edm::ConsumesCollector& iCC)
    : VertexTimeAlgorithmBase(iConfig, iCC) {}

void VertexTimeAlgorithmLegacy4D::fillPSetDescription(edm::ParameterSetDescription& iDesc) {
  VertexTimeAlgorithmBase::fillPSetDescription(iDesc);
}

void VertexTimeAlgorithmLegacy4D::setEvent(edm::Event& iEvent, edm::EventSetup const&){};

bool VertexTimeAlgorithmLegacy4D::vertexTime(float& vtxTime, float& vtxTimeError, const TransientVertex& vtx) const {
  const auto num_track = vtx.originalTracks().size();
  if (num_track == 0) {
    return false;
  }

  double sumwt = 0.;
  double sumwt2 = 0.;
  double sumw = 0.;
  double vartime = 0.;

  for (const auto& trk : vtx.originalTracks()) {
    const double time = trk.timeExt();
    const double err = trk.dtErrorExt();
    if ((time == 0) && (err > TransientTrackBuilder::defaultInvalidTrackTimeReso))
      continue;  // tracks with no time information, as implemented in TransientTrackBuilder.cc l.17
    const double inverr = err > 0. ? 1.0 / err : 0.;
    const double w = inverr * inverr;
    sumwt += w * time;
    sumwt2 += w * time * time;
    sumw += w;
  }

  if (sumw > 0) {
    double sumsq = sumwt2 - sumwt * sumwt / sumw;
    double chisq = num_track > 1 ? sumsq / double(num_track - 1) : sumsq / double(num_track);
    vartime = chisq / sumw;

    vtxTime = sumwt / sumw;
    vtxTimeError = sqrt(vartime);
    return true;
  }

  vtxTime = 0;
  vtxTimeError = 1.;
  return false;
}

#include "AreaSeededTrackingRegionsBuilder.h"

#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Math/interface/PtEtaPhiMass.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/normalizedPhi.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTrackerEvent.h"

#include <array>
#include <limits>

namespace {
  float perp2(const std::array<float, 2>& a) { return a[0] * a[0] + a[1] * a[1]; }
}  // namespace

AreaSeededTrackingRegionsBuilder::AreaSeededTrackingRegionsBuilder(const edm::ParameterSet& regPSet,
                                                                   edm::ConsumesCollector& iC)
    : candidates_(regPSet, iC) {
  m_extraPhi = regPSet.getParameter<double>("extraPhi");
  m_extraEta = regPSet.getParameter<double>("extraEta");

  // RectangularEtaPhiTrackingRegion parameters:
  m_ptMin = regPSet.getParameter<double>("ptMin");
  m_originRadius = regPSet.getParameter<double>("originRadius");
  m_precise = regPSet.getParameter<bool>("precise");
  m_whereToUseMeasurementTracker = RectangularEtaPhiTrackingRegion::stringToUseMeasurementTracker(
      regPSet.getParameter<std::string>("whereToUseMeasurementTracker"));
  if (m_whereToUseMeasurementTracker != RectangularEtaPhiTrackingRegion::UseMeasurementTracker::kNever) {
    token_measurementTracker =
        iC.consumes<MeasurementTrackerEvent>(regPSet.getParameter<edm::InputTag>("measurementTrackerName"));
  }
  m_searchOpt = regPSet.getParameter<bool>("searchOpt");
}

void AreaSeededTrackingRegionsBuilder::fillDescriptions(edm::ParameterSetDescription& desc) {
  desc.add<double>("extraPhi", 0.);
  desc.add<double>("extraEta", 0.);

  desc.add<double>("ptMin", 0.9);
  desc.add<double>("originRadius", 0.2);
  desc.add<bool>("precise", true);

  desc.add<std::string>("whereToUseMeasurementTracker", "Never");
  desc.add<edm::InputTag>("measurementTrackerName", edm::InputTag(""));
  TrackingSeedCandidates::fillDescriptions(desc);
  desc.add<bool>("searchOpt", false);
}

AreaSeededTrackingRegionsBuilder::Builder AreaSeededTrackingRegionsBuilder::beginEvent(const edm::Event& e) const {
  auto builder = Builder(this);

  if (!token_measurementTracker.isUninitialized()) {
    edm::Handle<MeasurementTrackerEvent> hmte;
    e.getByToken(token_measurementTracker, hmte);
    builder.setMeasurementTracker(hmte.product());
  }
  builder.setCandidates((candidates_.objects(e)));
  return builder;
}

std::vector<std::unique_ptr<TrackingRegion> > AreaSeededTrackingRegionsBuilder::Builder::regions(
    const Origins& origins, const std::vector<Area>& areas) const {
  std::vector<std::unique_ptr<TrackingRegion> > result;

  // create tracking regions in directions of the points of interest
  int n_regions = 0;
  for (const auto& origin : origins) {
    auto reg = region(origin, areas);
    if (!reg)
      continue;
    result.push_back(std::move(reg));
    ++n_regions;
  }
  LogDebug("AreaSeededTrackingRegionsBuilder") << "produced " << n_regions << " regions";

  return result;
}

std::unique_ptr<TrackingRegion> AreaSeededTrackingRegionsBuilder::Builder::region(
    const Origin& origin, const std::vector<Area>& areas) const {
  return regionImpl(origin, areas);
}
std::unique_ptr<TrackingRegion> AreaSeededTrackingRegionsBuilder::Builder::region(
    const Origin& origin, const edm::VecArray<Area, 2>& areas) const {
  return regionImpl(origin, areas);
}

template <typename T>
std::unique_ptr<TrackingRegion> AreaSeededTrackingRegionsBuilder::Builder::regionImpl(const Origin& origin,
                                                                                      const T& areas) const {
  float minEta = std::numeric_limits<float>::max(), maxEta = std::numeric_limits<float>::lowest();
  float minPhi = std::numeric_limits<float>::max(), maxPhi = std::numeric_limits<float>::lowest();

  const auto& orig = origin.first;

  LogDebug("AreaSeededTrackingRegionsProducer") << "Origin x,y,z " << orig.x() << "," << orig.y() << "," << orig.z();

  auto vecFromOrigPlusRadius = [&](const std::array<float, 2>& vec) {
    const auto invlen = 1.f / std::sqrt(perp2(vec));
    const auto tmp = (1.f - m_conf->m_originRadius * invlen);
    return std::array<float, 2>{{vec[0] * tmp, vec[1] * tmp}};
  };
  auto tangentVec = [&](const std::array<float, 2>& vec, int sign) {
    const auto d = std::sqrt(perp2(vec));
    const auto r = m_conf->m_originRadius;
    const auto tmp = r / std::sqrt(d * d - r * r);
    // sign+ for counterclockwise, sign- for clockwise
    const auto orthvec = sign > 0 ? std::array<float, 2>{{-vec[1] * tmp, vec[0] * tmp}}
                                  : std::array<float, 2>{{vec[1] * tmp, -vec[0] * tmp}};
    return std::array<float, 2>{{vec[0] - orthvec[0], vec[1] - orthvec[1]}};
  };

  for (const auto& area : areas) {
    // straight line assumption is conservative, accounding for
    // low-pT bending would only tighten the eta-phi window
    LogTrace("AreaSeededTrackingRegionsBuilder")
        << " area x,y points " << area.x_rmin_phimin() << "," << area.y_rmin_phimin() << " "  // go around
        << area.x_rmin_phimax() << "," << area.y_rmin_phimax() << " " << area.x_rmax_phimax() << ","
        << area.y_rmax_phimax() << " " << area.x_rmax_phimin() << "," << area.y_rmax_phimin() << " "
        << "z " << area.zmin() << "," << area.zmax();

    // some common variables
    const float x_rmin_phimin = area.x_rmin_phimin() - orig.x();
    const float x_rmin_phimax = area.x_rmin_phimax() - orig.x();
    const float y_rmin_phimin = area.y_rmin_phimin() - orig.y();
    const float y_rmin_phimax = area.y_rmin_phimax() - orig.y();

    const std::array<float, 2> p_rmin_phimin = {{x_rmin_phimin, y_rmin_phimin}};
    const std::array<float, 2> p_rmin_phimax = {{x_rmin_phimax, y_rmin_phimax}};
    const std::array<float, 2> p_rmax_phimin = {{area.x_rmax_phimin() - orig.x(), area.y_rmax_phimin() - orig.y()}};
    const std::array<float, 2> p_rmax_phimax = {{area.x_rmax_phimax() - orig.x(), area.y_rmax_phimax() - orig.y()}};

    // eta
    {
      // find which of the two rmin points is closer to the origin
      //
      // closest point for p_rmin, farthest point for p_rmax
      const std::array<float, 2> p_rmin = perp2(p_rmin_phimin) < perp2(p_rmin_phimax) ? p_rmin_phimin : p_rmin_phimax;
      const std::array<float, 2> p_rmax = perp2(p_rmax_phimin) > perp2(p_rmax_phimax) ? p_rmax_phimin : p_rmax_phimax;

      // then calculate the xy vector from the point closest to p_rmin on
      // the (origin,originRadius) circle to the p_rmin
      // this will maximize the eta window
      const auto p_min = vecFromOrigPlusRadius(p_rmin);
      const auto p_max = vecFromOrigPlusRadius(p_rmax);

      // then we calculate the maximal eta window
      const auto etamin = std::min(etaFromXYZ(p_min[0], p_min[1], area.zmin() - (orig.z() + origin.second)),
                                   etaFromXYZ(p_max[0], p_max[1], area.zmin() - (orig.z() + origin.second)));
      const auto etamax = std::max(etaFromXYZ(p_min[0], p_min[1], area.zmax() - (orig.z() - origin.second)),
                                   etaFromXYZ(p_max[0], p_max[1], area.zmax() - (orig.z() - origin.second)));

      LogTrace("AreaSeededTrackingRegionBuilder") << "  eta min,max " << etamin << "," << etamax;

      minEta = std::min(minEta, etamin);
      maxEta = std::max(maxEta, etamax);
    }

    // phi
    {
      // For phi we construct the tangent lines of (origin,
      // originRadius) that go though each of the 4 points (in xy
      // plane) of the area. Easiest is to make a vector orthogonal to
      // origin->point vector which has a length of
      //
      // length = r*d/std::sqrt(d*d-r*r)
      //
      // where r is the originRadius and d is the distance from origin
      // to the point (to derive draw the situation and start with
      // definitions of sin/cos of one of the angles of the
      // right-angled triangle.

      // But we start with a "reference phi" so that we can easily
      // decide which phi is the largest/smallest without having too
      // much of headache with the wrapping. The reference is in
      // principle defined by the averages of y&x phimin/phimax at
      // rmin, but the '0.5f*' factor is omitted here to reduce
      // computations.
      const auto phi_ref = std::atan2(y_rmin_phimin + y_rmin_phimax, x_rmin_phimin + x_rmin_phimax);

      // for maximum phi we need the orthogonal vector to the left
      const auto tan_rmin_phimax = tangentVec(p_rmin_phimax, +1);
      const auto tan_rmax_phimax = tangentVec(p_rmax_phimax, +1);
      const auto phi_rmin_phimax = std::atan2(tan_rmin_phimax[1], tan_rmin_phimax[0]);
      const auto phi_rmax_phimax = std::atan2(tan_rmax_phimax[1], tan_rmax_phimax[0]);

      auto phimax =
          std::abs(reco::deltaPhi(phi_rmin_phimax, phi_ref)) > std::abs(reco::deltaPhi(phi_rmax_phimax, phi_ref))
              ? phi_rmin_phimax
              : phi_rmax_phimax;

      LogTrace("AreaSeededTrackingRegionBuilder")
          << "   rmin_phimax vec " << p_rmin_phimax[0] << "," << p_rmin_phimax[1] << " tangent " << tan_rmin_phimax[0]
          << "," << tan_rmin_phimax[1] << " phi " << phi_rmin_phimax << "\n"
          << "   rmax_phimax vec " << p_rmax_phimax[0] << "," << p_rmax_phimax[1] << " tangent " << tan_rmax_phimax[0]
          << "," << tan_rmax_phimax[1] << " phi " << phi_rmax_phimax << "\n"
          << "   phimax " << phimax;

      // for minimum phi we need the orthogonal vector to the right
      const auto tan_rmin_phimin = tangentVec(p_rmin_phimin, -1);
      const auto tan_rmax_phimin = tangentVec(p_rmax_phimin, -1);
      const auto phi_rmin_phimin = std::atan2(tan_rmin_phimin[1], tan_rmin_phimin[0]);
      const auto phi_rmax_phimin = std::atan2(tan_rmax_phimin[1], tan_rmax_phimin[0]);

      auto phimin =
          std::abs(reco::deltaPhi(phi_rmin_phimin, phi_ref)) > std::abs(reco::deltaPhi(phi_rmax_phimin, phi_ref))
              ? phi_rmin_phimin
              : phi_rmax_phimin;

      LogTrace("AreaSeededTrackingRegionBuilder")
          << "   rmin_phimin vec " << p_rmin_phimin[0] << "," << p_rmin_phimin[1] << " tangent " << tan_rmin_phimin[0]
          << "," << tan_rmin_phimin[1] << " phi " << phi_rmin_phimin << "\n"
          << "   rmax_phimin vec " << p_rmax_phimin[0] << "," << p_rmax_phimin[1] << " tangent " << tan_rmax_phimin[0]
          << "," << tan_rmax_phimin[1] << " phi " << phi_rmax_phimin << "\n"
          << "   phimin " << phimin;

      // wrapped around, need to decide which one to wrap
      if (phimax < phimin) {
        if (phimax < 0)
          phimax += 2 * M_PI;
        else
          phimin -= 2 * M_PI;
      }

      LogTrace("AreaSeededTrackingRegionBuilder") << "  phi min,max " << phimin << "," << phimax;

      minPhi = std::min(minPhi, phimin);
      maxPhi = std::max(maxPhi, phimax);
    }

    LogTrace("AreaSeededTrackingRegionBuilder")
        << "  windows after this area  eta " << minEta << "," << maxEta << " phi " << minPhi << "," << maxPhi;
  }

  const auto meanEta = (minEta + maxEta) / 2.f;
  const auto meanPhi = (minPhi + maxPhi) / 2.f;
  const auto dEta = maxEta - meanEta + m_conf->m_extraEta;
  const auto dPhi = maxPhi - meanPhi + m_conf->m_extraPhi;

  auto useCandidates = false;
  if (candidates.first)
    useCandidates = true;

  if (useCandidates) {
    // If we have objects used for seeding, loop over objects and find overlap with the found region. Return overlaps as tracking regions to use
    for (const auto& object : *candidates.first) {
      float dEta_Cand = candidates.second.first;
      float dPhi_Cand = candidates.second.second;
      float eta_Cand = object.eta();
      float phi_Cand = object.phi();
      float dEta_Cand_Point = std::abs(eta_Cand - meanEta);
      float dPhi_Cand_Point = std::abs(deltaPhi(phi_Cand, meanPhi));

      if (dEta_Cand_Point > (dEta_Cand + dEta) || dPhi_Cand_Point > (dPhi_Cand + dPhi))
        continue;

      float etaMin_RoI = std::max(eta_Cand - dEta_Cand, meanEta - dEta);
      float etaMax_RoI = std::min(eta_Cand + dEta_Cand, meanEta + dEta);

      float phi_Cand_minus = normalizedPhi(phi_Cand - dPhi_Cand);
      float phi_Point_minus = normalizedPhi(meanPhi - dPhi);
      float phi_Cand_plus = normalizedPhi(phi_Cand + dPhi_Cand);
      float phi_Point_plus = normalizedPhi(meanPhi + dPhi);

      float phiMin_RoI = deltaPhi(phi_Cand_minus, phi_Point_minus) > 0. ? phi_Cand_minus : phi_Point_minus;
      float phiMax_RoI = deltaPhi(phi_Cand_plus, phi_Point_plus) < 0. ? phi_Cand_plus : phi_Point_plus;

      const auto meanEtaTemp = (etaMin_RoI + etaMax_RoI) / 2.f;
      auto meanPhiTemp = (phiMin_RoI + phiMax_RoI) / 2.f;
      if (phiMax_RoI < phiMin_RoI)
        meanPhiTemp -= M_PI;
      meanPhiTemp = normalizedPhi(meanPhiTemp);

      const auto dPhiTemp = deltaPhi(phiMax_RoI, meanPhiTemp);
      const auto dEtaTemp = etaMax_RoI - meanEtaTemp;

      const auto x = std::cos(meanPhiTemp);
      const auto y = std::sin(meanPhiTemp);
      const auto z = 1. / std::tan(2.f * std::atan(std::exp(-meanEtaTemp)));

      LogTrace("AreaSeededTrackingRegionsBuilder")
          << "Direction x,y,z " << x << "," << y << "," << z << " eta,phi " << meanEtaTemp << "," << meanPhiTemp
          << " window eta " << (meanEtaTemp - dEtaTemp) << "," << (meanEtaTemp + dEtaTemp) << " phi "
          << (meanPhiTemp - dPhiTemp) << "," << (meanPhiTemp + dPhiTemp);

      return std::make_unique<RectangularEtaPhiTrackingRegion>(GlobalVector(x, y, z),
                                                               origin.first,  // GlobalPoint
                                                               m_conf->m_ptMin,
                                                               m_conf->m_originRadius,
                                                               origin.second,
                                                               dEtaTemp,
                                                               dPhiTemp,
                                                               m_conf->m_whereToUseMeasurementTracker,
                                                               m_conf->m_precise,
                                                               m_measurementTracker,
                                                               m_conf->m_searchOpt);
    }
    // Have to retun nullptr here to ensure that we always return something
    return nullptr;

  } else {
    const auto x = std::cos(meanPhi);
    const auto y = std::sin(meanPhi);
    const auto z = 1. / std::tan(2.f * std::atan(std::exp(-meanEta)));

    LogTrace("AreaSeededTrackingRegionsBuilder")
        << "Direction x,y,z " << x << "," << y << "," << z << " eta,phi " << meanEta << "," << meanPhi << " window eta "
        << (meanEta - dEta) << "," << (meanEta + dEta) << " phi " << (meanPhi - dPhi) << "," << (meanPhi + dPhi);

    return std::make_unique<RectangularEtaPhiTrackingRegion>(GlobalVector(x, y, z),
                                                             origin.first,  // GlobalPoint
                                                             m_conf->m_ptMin,
                                                             m_conf->m_originRadius,
                                                             origin.second,
                                                             dEta,
                                                             dPhi,
                                                             m_conf->m_whereToUseMeasurementTracker,
                                                             m_conf->m_precise,
                                                             m_measurementTracker,
                                                             m_conf->m_searchOpt);
  }
}

#include "ForwardDiskSectorBuilderFromWedges.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/GeometryVector/interface/VectorUtil.h"

using namespace std;

// Warning, remember to assign this pointer to a ReferenceCountingPointer!
BoundDiskSector* ForwardDiskSectorBuilderFromWedges::operator()(const vector<const TECWedge*>& wedges) const {
  // check first that all wedges are about at the same phi and z !
  float phiStart = wedges.front()->position().phi();
  float zStart = wedges.front()->position().z();
  float wphimin, wphimax;
  for (auto wedge : wedges) {
    float zdiff = (*wedge).surface().position().z() - zStart;
    if (std::abs(zdiff) > 5.)
      edm::LogError("TkDetLayers") << " ForwardDiskSectorBuilderFromWedges: Trying to build "
                                   << "Petal from Wedges at different z ! Delta Z = " << zdiff;
    float wphi = (*wedge).surface().position().phi();
    if (Geom::phiLess(phiStart, wphi)) {
      wphimin = phiStart;
      wphimax = wphi;
    } else {
      wphimin = wphi;
      wphimax = phiStart;
    }
    float phidiff = wphimax - wphimin;
    if (phidiff < 0.)
      phidiff += 2. * Geom::pi();
    if (phidiff > 0.3)
      edm::LogError("TkDetLayers") << " ForwardDiskSectorBuilderFromWedges: Trying to build "
                                   << "Petal from Wedges at different phi ! Delta phi = " << phidiff;
  }

  auto bo = computeBounds(wedges);

  Surface::PositionType pos(bo.second.x(), bo.second.y(), bo.second.z());
  Surface::RotationType rot = computeRotation(wedges, pos);
  return new BoundDiskSector(pos, rot, bo.first);
}

pair<DiskSectorBounds*, GlobalVector> ForwardDiskSectorBuilderFromWedges::computeBounds(
    const vector<const TECWedge*>& wedges) const {
  // compute maximum and minimum radius and phi
  float rmin((**(wedges.begin())).specificSurface().innerRadius());
  float rmax(rmin);
  float zmin((**(wedges.begin())).surface().position().z());
  float zmax(zmin);
  float phimin((**(wedges.begin())).surface().position().phi());
  float phimax(phimin);

  for (auto wedge : wedges) {
    // edm::LogInfo(TkDetLayers) << "---------------------------------------------" ;
    // edm::LogInfo(TkDetLayers) <<   " Builder: Position of wedge     :" << (**iw).position() ;
    float ri = (*wedge).specificSurface().innerRadius();
    float ro = (*wedge).specificSurface().outerRadius();
    float zmi = (*wedge).surface().position().z() - (*wedge).specificSurface().bounds().thickness() / 2.;
    float zma = (*wedge).surface().position().z() + (*wedge).specificSurface().bounds().thickness() / 2.;
    float phi1 = (*wedge).surface().position().phi() - (*wedge).specificSurface().phiHalfExtension();
    float phi2 = (*wedge).surface().position().phi() + (*wedge).specificSurface().phiHalfExtension();
    rmin = min(rmin, ri);
    rmax = max(rmax, ro);
    zmin = min(zmin, zmi);
    zmax = max(zmax, zma);
    if (Geom::phiLess(phi1, phimin))
      phimin = phi1;
    if (Geom::phiLess(phimax, phi2))
      phimax = phi2;
  }

  if (!Geom::phiLess(phimin, phimax))
    edm::LogError("TkDetLayers") << " ForwardDiskSectorBuilderFromWedges : "
                                 << "Something went wrong with Phi Sorting !";
  float zPos = (zmax + zmin) / 2.;
  float phiWin = phimax - phimin;
  float phiPos = (phimax + phimin) / 2.;
  float rmed = (rmin + rmax) / 2.;
  if (phiWin < 0.) {
    if ((phimin < Geom::pi() / 2.) || (phimax > -Geom::pi() / 2.)) {
      edm::LogError("TkDetLayers") << " Debug: something strange going on, please check ";
    }
    // edm::LogInfo(TkDetLayers) << " Petal at pi: phi " << phimin << " " << phimax << " " << phiWin
    //	 << " " << 2.*Geom::pi()+phiWin << " " ;
    phiWin += 2. * Geom::pi();
    phiPos += Geom::pi();
  }

  GlobalVector pos(rmed * cos(phiPos), rmed * sin(phiPos), zPos);
  return make_pair(new DiskSectorBounds(rmin, rmax, zmin - zPos, zmax - zPos, phiWin), pos);
}

Surface::RotationType ForwardDiskSectorBuilderFromWedges::computeRotation(const vector<const TECWedge*>& wedges,
                                                                          Surface::PositionType pos) const {
  GlobalVector yAxis = (GlobalVector(pos.x(), pos.y(), 0.)).unit();

  GlobalVector zAxis(0., 0., 1.);
  GlobalVector xAxis = yAxis.cross(zAxis);

  return Surface::RotationType(xAxis, yAxis);
}

#include "Validation/MuonGEMHits/interface/BaseMatcher.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"


BaseMatcher::BaseMatcher(const SimTrack& t, const SimVertex& v,
      const edm::ParameterSet& ps, const edm::Event& ev, const edm::EventSetup& es)
: trk_(t), vtx_(v), conf_(ps), ev_(ev), es_(es), verbose_(0)
{
  // Get the magnetic field
  es.get< IdealMagneticFieldRecord >().get(magfield_);

  // Get the propagators                                                                                  
  es.get< TrackingComponentsRecord >().get("SteppingHelixPropagatorAlong", propagator_);
  es.get< TrackingComponentsRecord >().get("SteppingHelixPropagatorOpposite", propagatorOpposite_);

  try{
    es.get<MuonGeometryRecord>().get(gem_geom);
  }
  catch( edm::eventsetup::NoProxyException<GEMGeometry>& e) {
      edm::LogError("BaseMatcher") << "+++ Error : GEM geometry is unavailable. +++\n";
      return;
  }
  

}


BaseMatcher::~BaseMatcher()
{
}

GlobalPoint
BaseMatcher::propagateToZ(GlobalPoint &inner_point, GlobalVector &inner_vec, float z) const
{
  Plane::PositionType pos(0.f, 0.f, z);
  Plane::RotationType rot;
  Plane::PlanePointer my_plane(Plane::build(pos, rot));

  FreeTrajectoryState state_start(inner_point, inner_vec, trk_.charge(), &*magfield_);

  TrajectoryStateOnSurface tsos(propagator_->propagate(state_start, *my_plane));
  if (!tsos.isValid()) tsos = propagatorOpposite_->propagate(state_start, *my_plane);

  if (tsos.isValid()) return tsos.globalPosition();
  return GlobalPoint();
}


GlobalPoint
BaseMatcher::propagateToZ(float z) const
{
  GlobalPoint inner_point(vtx_.position().x(), vtx_.position().y(), vtx_.position().z());
  GlobalVector inner_vec (trk_.momentum().x(), trk_.momentum().y(), trk_.momentum().z());
  return propagateToZ(inner_point, inner_vec, z);
}

GlobalPoint
BaseMatcher::propagatedPositionGEM(int station=1) const
{
  const double eta(trk().momentum().eta());
  const int endcap( (eta > 0.) ? 1 : -1);
  const GEMGeometry* GEMGeometry = &*gem_geom;
  GEMDetId* dummy = new GEMDetId( (int)endcap, 1, (int)station, 1, 1,  1);
  GEMDetId* dummy2 = new GEMDetId( (int)endcap, 1, (int)station, 2, 1, 1);
  const LocalPoint p0(0,0,0);
  const GlobalPoint g_p = GEMGeometry->idToDet(dummy->rawId())->surface().toGlobal(p0);
  const GlobalPoint g_p2 = GEMGeometry->idToDet(dummy2->rawId())->surface().toGlobal(p0);
  const double z = (TMath::Abs( g_p.z() + g_p2.z()))/2. ;

  return propagateToZ(endcap*z);
}

#include "TrackingTools/IPTools/interface/ImpactParameterComputer.h"

IPTools::ImpactParameterComputer::ImpactParameterComputer(const reco::Vertex vtx) {
  _VtxPos    = RecoVertex::convertPos(vtx.position());
  _VtxPosErr = RecoVertex::convertError(vtx.error());
}
  
IPTools::ImpactParameterComputer::ImpactParameterComputer(const reco::BeamSpot bsp) {
  _VtxPos    = RecoVertex::convertPos(bsp.position());
  _VtxPosErr = RecoVertex::convertError(bsp.covariance3D());
}

IPTools::ImpactParameterComputer::~ImpactParameterComputer() { }

Measurement1D IPTools::ImpactParameterComputer::computeIP(const edm::EventSetup& es, const reco::Track tr, bool return3D){

  //get the transient track (e.g. for the muon track)
  edm::ESHandle<TransientTrackBuilder> builder;
  es.get<TransientTrackRecord>().get("TransientTrackBuilder", builder);
  reco::TransientTrack transTrack = builder->build(tr);

  ///define the position of the refPoint on the muon track and its error
  TrajectoryStateOnSurface tsos = IPTools::transverseExtrapolate(transTrack.impactPointState(), _VtxPos, transTrack.field());
  GlobalPoint refPoint          = tsos.globalPosition();
  GlobalError refPointErr       = tsos.cartesianError().position();
   
  Measurement1D mess1D(0.,0.);
  ///calculate the distance of the PV to the refPoint on the track
  if(!return3D){
    VertexDistanceXY distanceComputer;
    mess1D = distanceComputer.distance(VertexState(_VtxPos, _VtxPosErr), VertexState(refPoint, refPointErr));
  }
  else{ 
    VertexDistance3D distanceComputer; 
    mess1D = distanceComputer.distance(VertexState(_VtxPos, _VtxPosErr), VertexState(refPoint, refPointErr));
  }
  
  return mess1D;
}

Measurement1D IPTools::ImpactParameterComputer::computeIPdz(const edm::EventSetup& es, const reco::Track tr){

  //get the transient track (e.g. for the muon track)
  edm::ESHandle<TransientTrackBuilder> builder;
  es.get<TransientTrackRecord>().get("TransientTrackBuilder", builder);
  reco::TransientTrack transTrack = builder->build(tr);

  ///define the position of the refPoint on the muon track and its error
  TrajectoryStateOnSurface tsos = IPTools::transverseExtrapolate(transTrack.impactPointState(), _VtxPos, transTrack.field());
  GlobalPoint refPoint          = tsos.globalPosition();
  GlobalError refPointErr       = tsos.cartesianError().position();

  ///calculate the distance of the PV to the refPoint on the track
  double VtxPosZ, VtxPosZErr, RefPointZ, RefPointZErr;
  VtxPosZ      = _VtxPos.z();
  VtxPosZErr   = _VtxPosErr.czz();
  RefPointZ    = refPoint.z();
  RefPointZErr = refPointErr.czz();

  double dz    = ( VtxPosZ - RefPointZ );
  double dzErr = sqrt( VtxPosZErr*VtxPosZErr  + RefPointZErr*RefPointZErr);

  Measurement1D mess1D(dz, dzErr);
  return mess1D;
}


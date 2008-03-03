#include "Validation/RecoMuon/src/MuonSimRecoMatching.h"

#include "DataFormats/Math/interface/deltaR.h"

#include "MagneticField/Engine/interface/MagneticField.h" 
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h" 
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/PatternTools/interface/TSCPBuilderNoMaterial.h"

#include <algorithm>

using namespace std;
using namespace edm;
using namespace reco;

typedef TrajectoryStateOnSurface TSOS;
typedef TrackingParticleCollection TPColl;
typedef TPColl::const_iterator TPCIter;
typedef MuonCollection MuColl;
typedef MuColl::const_iterator MuCIter;

MuonDeltaR::MuonDeltaR(const double maxDeltaR):
  maxDeltaR_(maxDeltaR)
{
}

bool MuonDeltaR::operator()(const TrackingParticle& simPtcl) const
{
  if ( abs(simPtcl.pdgId()) != 13 ) return false;
  if ( simPtcl.pt() <= 0 ) return false;
  return true;
}

bool MuonDeltaR::operator()(const Muon& recoMuon) const
{
  return true;
}

bool MuonDeltaR::operator()(const TrackingParticle& simPtcl, 
                          const Muon& recoMuon, 
                          double& result) const  
{
  
  result = deltaR<double>(simPtcl.eta(), simPtcl.phi(), 
                          recoMuon.eta(), recoMuon.phi());
  if ( result > maxDeltaR_ ) return false;
  return true;
}

// MuTrkchi2 : 
// code snipped from SimTracker/TrackAssociation/src/TrackAssociatorByChi2.cc

/*
// MuTrkChi2::MuTrkChi2(const double maxChi2, 
//                      const EventSetup& eventSetup,
//                      const bool onlyDiagonal):
//                      maxChi2_(maxChi2),
//   onlyDiagonal_(onlyDiagonal)
// {
//   eventSetup.get<IdealMagneticFieldRecord>().get(theMF);
// }

// bool MuTrkChi2::operator()(TPCIter simPtcl, MuCIter recoMuon, 
//                            double& result) const
// {
//   if ( abs(simPtcl->pdgId()) != 13 ) return false;
//   if ( sqrt(simPtcl->momentum().perp2()) < 0.5 ) return false;
  
//   // FIXME //
//   TrackBase::ParameterVector recoParams;// = recoMuon->parameters();
//   TrackBase::CovarianceMatrix recoCovMatrix;// = recoMuon->covariance();
  
//   if ( onlyDiagonal_ ) {
//     for(unsigned int i=0; i<5; i++) {
//       for(unsigned int j=0; j<5; j++) {
//         if ( i!=j ) recoCovMatrix(i,j) = 0;
//       }
//     }
//   }
  
//   recoCovMatrix.Invert();
  
//   Basic3DVector<double> simMonAtVtx(simPtcl->momentum().x(),
//                                     simPtcl->momentum().y(),
//                                     simPtcl->momentum().z());
//   Basic3DVector<double> simVtx(simPtcl->vertex());
  
//   TrackBase::ParameterVector simParams;
//   bool testResult = paramsAtClosest(simVtx, simMonAtVtx, simPtcl->charge(),
//                                     simParams);
//   if ( ! testResult ) return false;
  
//   result = ROOT::Math::Similarity(recoParams - simParams, recoCovMatrix)/5;
  
//   if ( result > maxChi2_ ) return false;
  
//   return true;
// }

// bool MuTrkChi2::paramsAtClosest(const Basic3DVector<double> vtx,
//                                 const Basic3DVector<double> momAtVtx,
//                                 const float charge,
//                                  TrackBase::ParameterVector& trkParams) const 
// {
//   FreeTrajectoryState ftsAtProd(GlobalPoint(vtx.x(), vtx.y(), vtx.z()),
//                                 GlobalVector(momAtVtx.x(), momAtVtx.y(), momAtVtx.z()),
//                                 TrackCharge(charge),
//                                 theMF.product());
//   TSCPBuilderNoMaterial tscpBuilder;
//   TrajectoryStateClosestToPoint tsAtClosest 
//     = tscpBuilder(ftsAtProd, GlobalPoint(0,0,0));
//   GlobalPoint v = tsAtClosest.theState().position();
//   GlobalVector p = tsAtClosest.theState().momentum();
  
//   if ( p.mag() == 0 || p.perp() == 0 ) return false;
  
//   trkParams[0] = tsAtClosest.charge()/p.mag();
//   trkParams[1] = Geom::halfPi() - p.theta();
//   trkParams[2] = p.phi();
//   trkParams[3] = (-v.x()*sin(p.phi())+v.y()*cos(p.phi()));
//   trkParams[4] = v.z()*p.perp()/p.mag() - (v.x()*p.x()+v.y()*p.y())/p.perp() * p.z()/p.mag();
  
//   return true;
// }
*/

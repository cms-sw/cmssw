// -*- C++ -*-
//
// Package:    L1TrackTrigger
// Class:      L1TkElectronTrackMatchAlgo
// 
/**\class L1TkElectronTrackMatchAlgo 

 Description: Algorithm to match L1EGamma oject with L1Track candidates

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  S. Dutta and A. Modak
//         Created:  Wed Dec 4 12 11:55:35 CET 2013
// $Id$
//
//


// system include files
#include <memory>
#include <cmath>

#include "DataFormats/Math/interface/deltaPhi.h"
#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/L1TkElectronTrackMatchAlgo.h"
namespace L1TkElectronTrackMatchAlgo {
  // ------------ match EGamma and Track
  void doMatch(l1extra::L1EmParticleCollection::const_iterator egIter, const edm::Ptr< L1TkTrackType > & pTrk, double& dph, double&  dr, double& deta) {
    GlobalPoint egPos = L1TkElectronTrackMatchAlgo::calorimeterPosition(egIter->phi(), egIter->eta(), egIter->energy());
    dph  = L1TkElectronTrackMatchAlgo::deltaPhi(egPos, pTrk);
    dr   = L1TkElectronTrackMatchAlgo::deltaR(egPos, pTrk);
    deta = L1TkElectronTrackMatchAlgo::deltaEta(egPos, pTrk);
  }
  // ------------ match EGamma and Track
  void doMatch(const GlobalPoint& epos, const edm::Ptr< L1TkTrackType > & pTrk, double& dph, double&  dr, double& deta) {
    dph  = L1TkElectronTrackMatchAlgo::deltaPhi(epos, pTrk);
    dr   = L1TkElectronTrackMatchAlgo::deltaR(epos, pTrk);
    deta = L1TkElectronTrackMatchAlgo::deltaEta(epos, pTrk);
  }
  // --------------- calculate deltaR between Track and EGamma object
  double deltaPhi(const GlobalPoint& epos, const edm::Ptr< L1TkTrackType > & pTrk){
    double er = epos.perp();

    // Using track fit curvature
    //  double curv = 0.003 * magnetStrength * trk->getCharge()/ trk->getMomentum().perp(); 
    double curv = pTrk->getRInv();
    double x1 = (asin(er*curv/(2.0)));
    double phi1 = reco::deltaPhi(pTrk->getMomentum().phi(), epos.phi());

    double dif1 = phi1 - x1;
    double dif2 = phi1 + x1; 

    if (fabs(dif1) < fabs(dif2)) return dif1;
    else return dif2; 
  
  }
// --------------- calculate deltaPhi between Track and EGamma object                 
  double deltaR(const GlobalPoint& epos, const edm::Ptr< L1TkTrackType > & pTrk){
    double dPhi = fabs(reco::deltaPhi(epos.phi(), pTrk->getMomentum().phi()));
    double dEta = deltaEta(epos, pTrk);
    return sqrt(dPhi*dPhi + dEta*dEta);
  }
  // --------------- calculate deltaEta between Track and EGamma object                 
  double deltaEta(const GlobalPoint& epos, const edm::Ptr< L1TkTrackType > & pTrk){
    double corr_eta = 999.0;
    double er = epos.perp();
    double ez = epos.z();
    double z0 = pTrk->getPOCA().z()  ;
    double theta = 0.0;
    if (ez >= 0) theta = atan(er/fabs(ez-z0));
    else theta = M_PI - atan(er/fabs(ez-z0));
    corr_eta = -1.0 * log(tan(theta/2.0));
    double deleta = (corr_eta - pTrk->getMomentum().eta());
    return deleta;
  }
  // -------------- get Calorimeter position
  GlobalPoint calorimeterPosition(double phi, double eta, double e) {
    double x = 0.; 
    double y = 0.;
    double z = 0.;
    double depth = 0.89*(7.7+ log(e) );
    double theta = 2*atan(exp(-1*eta));
    double r = 0;
    if( fabs(eta) > 1.479 ) 
      { 
	double ecalZ = 315.4*fabs(eta)/eta;
	
	r = ecalZ / cos( 2*atan( exp( -1*eta ) ) ) + depth;
	x = r * cos( phi ) * sin( theta );
	y = r * sin( phi ) * sin( theta );
	z = r * cos( theta );
      }
    else
      {
	double rperp = 129.0;
	double zface =  sqrt( cos( theta ) * cos( theta ) /
			     ( 1 - cos( theta ) * cos( theta ) ) * 
			     rperp * rperp ) * fabs( eta ) / eta;  
	r = sqrt( rperp * rperp + zface * zface ) + depth;
	x = r * cos( phi ) * sin( theta );
	y = r * sin( phi ) * sin( theta );
	z = r * cos( theta );
      }
    GlobalPoint pos(x,y,z);
    return pos;
  }

}

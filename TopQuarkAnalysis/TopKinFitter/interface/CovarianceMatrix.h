#ifndef CovarianceMatrix_h
#define CovarianceMatrix_h

#include "DataFormats/PatCandidates/interface/PATObject.h"
#include "TopQuarkAnalysis/TopObjectResolutions/interface/MET.h"
#include "TopQuarkAnalysis/TopObjectResolutions/interface/Jet.h"
#include "TopQuarkAnalysis/TopObjectResolutions/interface/Muon.h"
#include "TopQuarkAnalysis/TopObjectResolutions/interface/Electron.h"
#include "TopQuarkAnalysis/TopKinFitter/interface/TtSemiLepKinFitter.h"

class CovarianceMatrix {
    
 public:
  
  CovarianceMatrix(){};
  ~CovarianceMatrix(){};
 
  template <class ObjectType>
    TMatrixD setupMatrix(const pat::PATObject<ObjectType>& object, TtSemiLepKinFitter::Param param, std::string resolutionProvider);
};


template <class ObjectType>
TMatrixD CovarianceMatrix::setupMatrix(const pat::PATObject<ObjectType>& object, TtSemiLepKinFitter::Param param, std::string resolutionProvider = "")
{
  TMatrixD CovM3 (3,3); CovM3.Zero();
  TMatrixD CovM4 (4,4); CovM4.Zero();
  TMatrixD* CovM = &CovM3;
  // This part is for pat objects with resolutions embedded
  if(object.hasKinResolution())
    {
      switch(param){
      case TtSemiLepKinFitter::kEtEtaPhi :
	CovM3(0,0) = pow(object.resolEt(resolutionProvider) , 2);
	if( dynamic_cast<const reco::MET*>(&object) )CovM3(1,1) = pow(9999., 2);
	else CovM3(1,1) = pow(object.resolEta(resolutionProvider), 2);
	CovM3(2,2) = pow(object.resolPhi(resolutionProvider), 2);
	CovM = &CovM3;
	break;
      case TtSemiLepKinFitter::kEtThetaPhi :
	CovM3(0,0) = pow(object.resolEt(resolutionProvider)   , 2);
	CovM3(1,1) = pow(object.resolTheta(resolutionProvider), 2);
	CovM3(2,2) = pow(object.resolPhi(resolutionProvider)  , 2);
	CovM = &CovM3;
	break;
      case TtSemiLepKinFitter::kEMom :
	CovM4(0,0) = pow(1, 2);
	CovM4(1,1) = pow(1, 2);
	CovM4(2,2) = pow(1, 2);
	CovM4(3,3) = pow(1, 2);
	CovM = &CovM4;
	break;
      }
    }
  // This part is for objects without resolutions embedded
  else
    {
      double pt = object.pt(), eta = object.eta();
      // if object is a jet
      if( dynamic_cast<const reco::Jet*>(&object) ) {
	res::HelperJet jetRes;
	switch(param){
	case TtSemiLepKinFitter::kEMom :
	  if(resolutionProvider == "bjet") {
	    CovM4(0,0) = pow(jetRes.a (pt, eta, res::HelperJet::kB  ), 2); 
	    CovM4(1,1) = pow(jetRes.b (pt, eta, res::HelperJet::kB  ), 2); 
	    CovM4(2,2) = pow(jetRes.c (pt, eta, res::HelperJet::kB  ), 2);
	    CovM4(3,3) = pow(jetRes.d (pt, eta, res::HelperJet::kB  ), 2);
	  }
	  else {
	    CovM4(0,0) = pow(jetRes.a (pt, eta, res::HelperJet::kUds), 2);
	    CovM4(1,1) = pow(jetRes.b (pt, eta, res::HelperJet::kUds), 2);
	    CovM4(2,2) = pow(jetRes.c (pt, eta, res::HelperJet::kUds), 2);
	    CovM4(3,3) = pow(jetRes.d (pt, eta, res::HelperJet::kUds), 2);
	  }
	  CovM = &CovM4;
	  break;
	case TtSemiLepKinFitter::kEtEtaPhi : 
	  if(resolutionProvider == "bjet") {
	    CovM3(0,0) = pow(jetRes.et (pt, eta, res::HelperJet::kB  ), 2); 
	    CovM3(1,1) = pow(jetRes.eta(pt, eta, res::HelperJet::kB  ), 2); 
	    CovM3(2,2) = pow(jetRes.phi(pt, eta, res::HelperJet::kB  ), 2);
	  }
	  else {
	    CovM3(0,0) = pow(jetRes.et (pt, eta, res::HelperJet::kUds), 2);
	    CovM3(1,1) = pow(jetRes.eta(pt, eta, res::HelperJet::kUds), 2);
	    CovM3(2,2) = pow(jetRes.phi(pt, eta, res::HelperJet::kUds), 2);
	  }
	  CovM = &CovM3;
	  break;
	case TtSemiLepKinFitter::kEtThetaPhi :
	  if(resolutionProvider == "bjet") {
	    CovM3(0,0) = pow(jetRes.et   (pt, eta, res::HelperJet::kB  ), 2); 
	    CovM3(1,1) = pow(jetRes.theta(pt, eta, res::HelperJet::kB  ), 2); 
	    CovM3(2,2) = pow(jetRes.phi  (pt, eta, res::HelperJet::kB  ), 2);
	  }
	  else {
	    CovM3(0,0) = pow(jetRes.et   (pt, eta, res::HelperJet::kUds), 2);
	    CovM3(1,1) = pow(jetRes.theta(pt, eta, res::HelperJet::kUds), 2);
	    CovM3(2,2) = pow(jetRes.phi  (pt, eta, res::HelperJet::kUds), 2);
	  }
	  CovM = &CovM3;
	  break;
	} 
      }
      // if object is an electron
      else if( dynamic_cast<const reco::GsfElectron*>(&object) ) {
	res::HelperElectron elecRes;
	switch(param){
	case TtSemiLepKinFitter::kEMom :
	  CovM3(0,0) = pow(elecRes.a (pt, eta), 2);
	  CovM3(1,1) = pow(elecRes.b (pt, eta), 2); 
	  CovM3(2,2) = pow(elecRes.c (pt, eta), 2);
	  CovM = &CovM3;
	  break;
	case TtSemiLepKinFitter::kEtEtaPhi :
	  CovM3(0,0) = pow(elecRes.et (pt, eta), 2);
	  CovM3(1,1) = pow(elecRes.eta(pt, eta), 2); 
	  CovM3(2,2) = pow(elecRes.phi(pt, eta), 2);
	  CovM = &CovM3;
	  break;
	case TtSemiLepKinFitter::kEtThetaPhi :
	  CovM3(0,0) = pow(elecRes.et   (pt, eta), 2);
	  CovM3(1,1) = pow(elecRes.theta(pt, eta), 2); 
	  CovM3(2,2) = pow(elecRes.phi  (pt, eta), 2);
	  CovM = &CovM3;
	  break;
	}
      }
      // if object is a muon
      else if( dynamic_cast<const reco::Muon*>(&object) ) {
	res::HelperMuon muonRes;
	switch(param){
	case TtSemiLepKinFitter::kEMom :
	  CovM3(0,0) = pow(muonRes.a (pt, eta), 2);
	  CovM3(1,1) = pow(muonRes.b (pt, eta), 2); 
	  CovM3(2,2) = pow(muonRes.c (pt, eta), 2);
	  CovM = &CovM3;
	  break;
	case TtSemiLepKinFitter::kEtEtaPhi :
	  CovM3(0,0) = pow(muonRes.et (pt, eta), 2);
	  CovM3(1,1) = pow(muonRes.eta(pt, eta), 2); 
	  CovM3(2,2) = pow(muonRes.phi(pt, eta), 2);
	  CovM = &CovM3;
	  break;
	case TtSemiLepKinFitter::kEtThetaPhi :
	  CovM3(0,0) = pow(muonRes.et   (pt, eta), 2);
	  CovM3(1,1) = pow(muonRes.theta(pt, eta), 2); 
	  CovM3(2,2) = pow(muonRes.phi  (pt, eta), 2);
	  CovM = &CovM3;
	  break;
	}
      }
      // if object is met
      else if( dynamic_cast<const reco::MET*>(&object) ) {
	res::HelperMET metRes;
	switch(param){
	case TtSemiLepKinFitter::kEMom :
	  CovM3(0,0) = pow(metRes.a(pt), 2);
	  CovM3(1,1) = pow(metRes.b(pt), 2);
	  CovM3(2,2) = pow(metRes.c(pt), 2);
	  CovM = &CovM3;
	  break;
	case TtSemiLepKinFitter::kEtEtaPhi :
	  CovM3(0,0) = pow(metRes.et(pt), 2);
	  CovM3(1,1) = pow(          9999., 2);
	  CovM3(2,2) = pow(metRes.phi(pt), 2);
	  CovM = &CovM3;
	  break;
	case TtSemiLepKinFitter::kEtThetaPhi :
	  CovM3(0,0) = pow(metRes.et(pt), 2);
	  CovM3(1,1) = pow(          9999., 2);
	  CovM3(2,2) = pow(metRes.phi(pt), 2);
	  CovM = &CovM3;
	  break;
	}
      }
    }
  return *CovM;
}

#endif

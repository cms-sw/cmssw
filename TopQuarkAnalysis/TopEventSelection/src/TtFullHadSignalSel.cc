#include "TopQuarkAnalysis/TopEventSelection/interface/TtFullHadSignalSel.h"
#include "PhysicsTools/CandUtils/interface/EventShapeVariables.h"
#include "PhysicsTools/CandUtils/interface/Thrust.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "TLorentzVector.h"
#include "TVector3.h"

TtFullHadSignalSel::TtFullHadSignalSel()
{
}

std::vector<math::XYZVector> makeVecForEventShape(std::vector<pat::Jet> jets, bool only6Jets = true, ROOT::Math::Boost boost = ROOT::Math::Boost(0.,0.,0.)) {
  std::vector<math::XYZVector> p;
  bool doBoost = (boost == ROOT::Math::Boost(0.,0.,0.)) ? false : true;
  for(std::vector<pat::Jet>::const_iterator jet = jets.begin(); jet != jets.end(); ++jet){
    math::XYZVector Vjet;
    if(doBoost){
      ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > Ljet(jet->px(), jet->py(), jet->pz(), jet->energy());
      Vjet = math::XYZVector(boost(Ljet).Px(), boost(Ljet).Py(), boost(Ljet).Pz());
    }
    else
      Vjet = math::XYZVector(jet->px(), jet->py(), jet->pz());
    p.push_back(Vjet);
    if(only6Jets && jet-jets.begin()==5) break;
  }
  return p;
}

TtFullHadSignalSel::TtFullHadSignalSel(const std::vector<pat::Jet>& jets)
{

  H_      = -1.;
  Ht_     = -1.;
  Ht123_  = -1.;
  Ht3jet_ = -1.;
  Et56_   = -1.;
  sqrt_s_ = -1.;
  M3_     = -1.;
  
  TCHE_Bjets_   = 0.;
  TCHP_Bjets_   = 0.;
  SSVHE_Bjets_  = 0.;
  SSVHP_Bjets_  = 0.;
  CSV_Bjets_    = 0.;
  CSVMVA_Bjets_ = 0.;
  SM_Bjets_     = 0.;

  jets_etaetaMoment_ = 0.;
  jets_etaphiMoment_ = 0.;
  jets_phiphiMoment_ = 0.;

  jets_etaetaMomentLogEt_ = 0.;
  jets_etaphiMomentLogEt_ = 0.;
  jets_phiphiMomentLogEt_ = 0.;

  jets_etaetaMomentNoB_ = 0.;
  jets_etaphiMomentNoB_ = 0.;
  jets_phiphiMomentNoB_ = 0.;

  aplanarity_  = -1.;
  sphericity_  = -1.;
  circularity_ = -1.;
  isotropy_ = -1.;
  C_ = -1.;
  D_ = -1.;

  aplanarityAll_  = -1.;
  sphericityAll_  = -1.;
  circularityAll_ = -1.;
  isotropyAll_ = -1.;
  CAll_ = -1.;
  DAll_ = -1.;

  aplanarityAllCMS_  = -1.;
  sphericityAllCMS_  = -1.;
  circularityAllCMS_ = -1.;
  isotropyAllCMS_ = -1.;
  CAllCMS_ = -1.;
  DAllCMS_ = -1.;

  thrust_ = -1.;
  thrustCMS_ = -1.;

  TCHE_BJet_Discs_   = std::vector<double>(0);
  TCHP_BJet_Discs_   = std::vector<double>(0);
  SSVHE_BJet_Discs_  = std::vector<double>(0);
  SSVHP_BJet_Discs_  = std::vector<double>(0);
  CSV_BJet_Discs_    = std::vector<double>(0);
  CSVMVA_BJet_Discs_ = std::vector<double>(0);
  SM_BJet_Discs_     = std::vector<double>(0);

  pts_               = std::vector<double>(0);
  EtSin2Thetas_      = std::vector<double>(0);
  thetas_            = std::vector<double>(0);
  thetaStars_        = std::vector<double>(0);
  EtStars_           = std::vector<double>(0);

  EtSin2Theta3jet_  = 0.;
  theta3jet_        = 0.;
  thetaStar3jet_    = 0.;
  sinTheta3jet_     = 0.;
  sinThetaStar3jet_ = 0.;
  EtStar3jet_       = 0.;

  etaetaMoments_ = std::vector<double>(0);
  etaphiMoments_ = std::vector<double>(0);
  phiphiMoments_ = std::vector<double>(0);
 
  etaetaMomentsLogEt_ = std::vector<double>(0);
  etaphiMomentsLogEt_ = std::vector<double>(0);
  phiphiMomentsLogEt_ = std::vector<double>(0);
 
  etaetaMomentsMoment_ = std::vector<double>(0);
  etaphiMomentsMoment_ = std::vector<double>(0);
  phiphiMomentsMoment_ = std::vector<double>(0);
 
  etaetaMomentsMomentLogEt_ = std::vector<double>(0);
  etaphiMomentsMomentLogEt_ = std::vector<double>(0);
  phiphiMomentsMomentLogEt_ = std::vector<double>(0);
 
  etaetaMomentsNoB_ = std::vector<double>(0);
  etaphiMomentsNoB_ = std::vector<double>(0);
  phiphiMomentsNoB_ = std::vector<double>(0);
 
  dR_      = std::vector<double>(0);
  dRMass_  = std::vector<double>(0);
  dRAngle_ = std::vector<double>(0);

  dR3Jets_     = std::vector<double>(0);
  dR3JetsMass_ = std::vector<double>(0);

  massDiffMWCands_ = std::vector<double>(0);

  ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > totalSystem(0.,0.,0.,0.);

  std::vector< std::pair< double, std::vector<unsigned short> > > dRs(0);
  std::vector< std::pair< double, std::vector<unsigned short> > > dRs3Jets(0);

  std::vector< std::pair< double, std::vector<unsigned short> > > M3s(0);

  std::vector<ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > > fourVectors(0);

  unsigned short nonBJets = 0;
  for(std::vector<pat::Jet>::const_iterator jet = jets.begin(); jet != jets.end(); ++jet){

    H_      += jet->energy();
    Ht_     += jet->et();

    if(jet - jets.begin() < 3){
      Ht123_  += jet->et();
    }
    if(jet - jets.begin() == 4 || jet - jets.begin() == 5)
      Et56_ += jet->et();

    if(jet - jets.begin() > 1){
      Ht3jet_ += jet->et();
      EtSin2Theta3jet_ += jet->et()*pow(sin(jet->theta()),2);
      theta3jet_ += (jet->theta() > M_PI/2.)? (M_PI - jet->theta()) : jet->theta();
      sinTheta3jet_ += sin(jet->theta());
    }

    TCHE_BJet_Discs_  .push_back( jet->bDiscriminator("trackCountingHighEffBJetTags")         );
    TCHP_BJet_Discs_  .push_back( jet->bDiscriminator("trackCountingHighPurBJetTags")         );
    SSVHE_BJet_Discs_ .push_back( jet->bDiscriminator("simpleSecondaryVertexHighEffBJetTags") );
    SSVHP_BJet_Discs_ .push_back( jet->bDiscriminator("simpleSecondaryVertexHighPurBJetTags") );
    CSV_BJet_Discs_   .push_back( jet->bDiscriminator("combinedSecondaryVertexBJetTags")      );
    CSVMVA_BJet_Discs_.push_back( jet->bDiscriminator("combinedSecondaryVertexMVABJetTags")   );
    SM_BJet_Discs_    .push_back( jet->bDiscriminator("softMuonBJetTags")                     );
    
    pts_         .push_back(jet->pt());
    EtSin2Thetas_.push_back(jet->et()*pow(sin(jet->theta()),2));
    thetas_      .push_back( (jet->theta() > M_PI/2.)? (M_PI - jet->theta()) : jet->theta() );

    fourVectors.push_back(jet->p4());

    if(jet->bDiscriminator("trackCountingHighEffBJetTags")         > 3.3  ) ++TCHE_Bjets_;
    if(jet->bDiscriminator("trackCountingHighPurBJetTags")         > 3.41 ) ++TCHP_Bjets_;
    if(jet->bDiscriminator("simpleSecondaryVertexHighEffBJetTags") > 1.74 ) ++SSVHE_Bjets_;
    if(jet->bDiscriminator("simpleSecondaryVertexHighPurBJetTags") > 2.0  ) ++SSVHP_Bjets_;
    if(jet->bDiscriminator("combinedSecondaryVertexBJetTags")      > 0.75 ) ++CSV_Bjets_;
    if(jet->bDiscriminator("combinedSecondaryVertexMVABJetTags")   > 0.75 ) ++CSVMVA_Bjets_;
    if(jet->bDiscriminator("softMuonBJetTags")                     > 0.3  ) ++SM_Bjets_;
    
    if(jet->nConstituents() > 0){
      //if( jet->daughterPtr(0).productGetter()->getIt(jet->daughterPtr(0).id()) != 0 ){
	etaetaMoments_.push_back(         jet->etaetaMoment() );
	etaphiMoments_.push_back(std::abs(jet->etaphiMoment()));
	phiphiMoments_.push_back(         jet->phiphiMoment() );
	
	jets_etaetaMoment_ +=          jet->etaetaMoment();
	jets_etaphiMoment_ += std::abs(jet->etaphiMoment());
	jets_phiphiMoment_ +=          jet->phiphiMoment();

	etaetaMomentsLogEt_.push_back(         jet->etaetaMoment() *log(jet->et()));
	etaphiMomentsLogEt_.push_back(std::abs(jet->etaphiMoment())*log(jet->et()));
	phiphiMomentsLogEt_.push_back(         jet->phiphiMoment() *log(jet->et()));

	jets_etaetaMomentLogEt_ +=          jet->etaetaMoment() *log(jet->et());
	jets_etaphiMomentLogEt_ += std::abs(jet->etaphiMoment())*log(jet->et());
	jets_phiphiMomentLogEt_ +=          jet->phiphiMoment() *log(jet->et());
 
	if(jet->bDiscriminator("trackCountingHighEffBJetTags")         < 3.3  && jet->bDiscriminator("trackCountingHighPurBJetTags")         < 1.93 &&
	   jet->bDiscriminator("simpleSecondaryVertexHighEffBJetTags") < 1.74 && jet->bDiscriminator("simpleSecondaryVertexHighPurBJetTags") < 2.0  ){

	  ++nonBJets;

	  etaetaMomentsNoB_.push_back(         jet->etaetaMoment() );
	  etaphiMomentsNoB_.push_back(std::abs(jet->etaphiMoment()));
	  phiphiMomentsNoB_.push_back(         jet->phiphiMoment() );
	  
	  jets_etaetaMomentNoB_ +=          jet->etaetaMoment();
	  jets_etaphiMomentNoB_ += std::abs(jet->etaphiMoment());
	  jets_phiphiMomentNoB_ +=          jet->phiphiMoment();
	}
	//}
    }

    for(std::vector<pat::Jet>::const_iterator jet2 = jet+1; jet2 != jets.end(); ++jet2){
      unsigned short comb2A[2] = { (unsigned short)(jet-jets.begin()) , (unsigned short)(jet2-jets.begin()) };
      std::vector<unsigned short> comb2(comb2A, comb2A + sizeof(comb2A) / sizeof(unsigned short));
      dRs.push_back( std::make_pair( deltaR( jet->phi(), jet->eta(), jet2->phi(), jet2->eta() ), comb2 ) );

      for(std::vector<pat::Jet>::const_iterator jet3 = jet2+1; jet3 != jets.end(); ++jet3){
	unsigned short comb3A[3] = { (unsigned short)(jet-jets.begin()) , (unsigned short)(jet2-jets.begin()) , (unsigned short)(jet3-jets.begin()) };
	std::vector<unsigned short> comb3(comb3A, comb3A + sizeof(comb3A) / sizeof(unsigned short));
	double dR1 = deltaR( jet ->eta(), jet ->phi(), jet2->eta(), jet2->phi() );
	double dR2 = deltaR( jet ->eta(), jet ->phi(), jet3->eta(), jet3->phi() );
	double dR3 = deltaR( jet2->eta(), jet2->phi(), jet3->eta(), jet3->phi() );
	dRs3Jets.push_back( std::make_pair( dR1+dR2+dR3, comb3 ) );
	M3s.push_back( std::make_pair( ( jet->p4() + jet2->p4() + jet3->p4() ).pt(), comb3 ) );
      }
    }

    totalSystem += jet->p4();
  }

  ROOT::Math::Boost CoMBoostTotal(totalSystem.BoostToCM());
  std::vector<reco::LeafCandidate> boostedJets;

  for(std::vector<pat::Jet>::const_iterator jet = jets.begin(); jet != jets.end(); ++jet){
    boostedJets.push_back(reco::LeafCandidate(jet->charge(), CoMBoostTotal(jet->p4()), jet->vertex(), jet->pdgId(), jet->status(), true));
  }

  EtSin2Theta3jet_ /= ((double)(jets.size()-3));
  theta3jet_       /= ((double)(jets.size()-3));
  sinTheta3jet_    /= ((double)(jets.size()-3));

  jets_etaetaMoment_ /= (double)jets.size();
  jets_etaphiMoment_ /= (double)jets.size();
  jets_phiphiMoment_ /= (double)jets.size();

  jets_etaetaMomentLogEt_ /= (double)jets.size();
  jets_etaphiMomentLogEt_ /= (double)jets.size();
  jets_phiphiMomentLogEt_ /= (double)jets.size();

  if(nonBJets){
    jets_etaetaMomentNoB_ /= (double)nonBJets;
    jets_etaphiMomentNoB_ /= (double)nonBJets;
    jets_phiphiMomentNoB_ /= (double)nonBJets;
  }

  for(unsigned short i = 0 ; i < etaetaMoments_.size() ; ++i){
    etaetaMomentsMoment_.push_back(etaetaMoments_.at(i)/jets_etaetaMoment_);
    etaphiMomentsMoment_.push_back(etaphiMoments_.at(i)/jets_etaphiMoment_);
    phiphiMomentsMoment_.push_back(phiphiMoments_.at(i)/jets_phiphiMoment_);

    etaetaMomentsMomentLogEt_.push_back(etaetaMomentsLogEt_.at(i)/jets_etaetaMomentLogEt_);
    etaphiMomentsMomentLogEt_.push_back(etaphiMomentsLogEt_.at(i)/jets_etaphiMomentLogEt_);
    phiphiMomentsMomentLogEt_.push_back(phiphiMomentsLogEt_.at(i)/jets_phiphiMomentLogEt_);
  }

  std::sort(dRs     .begin(), dRs     .end());
  std::sort(dRs3Jets.begin(), dRs3Jets.end());
  
  for(std::vector< std::pair< double, std::vector<unsigned short> > >::const_iterator dR = dRs.begin(); dR != dRs.end(); ++dR){
    dR_.push_back(dR->first);
    dRMass_.push_back((jets.at(dR->second.at(0)).p4()+jets.at(dR->second.at(1)).p4()).mass());

    ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > wHypo = jets.at(dR->second.at(0)).p4()+jets.at(dR->second.at(1)).p4();
    TLorentzVector wHypoHelper(wHypo.Px(), wHypo.Py(), wHypo.Pz(), wHypo.E());
    wHypoHelper.SetVectM(TVector3(wHypo.Px(), wHypo.Py(), wHypo.Pz()), 80.4);
    wHypo.SetPxPyPzE(wHypoHelper.Px(), wHypoHelper.Py(), wHypoHelper.Pz(), wHypoHelper.E());
    ROOT::Math::Boost CoMBoostWHypo(wHypo.BoostToCM());
    dRAngle_.push_back(ROOT::Math::VectorUtil::Angle(CoMBoostWHypo(jets.at(dR->second.at(0)).p4()), CoMBoostWHypo(jets.at(dR->second.at(1)).p4())));
  }

  for(std::vector< std::pair< double, std::vector<unsigned short> > >::const_iterator dR = dRs3Jets.begin(); dR != dRs3Jets.end(); ++dR){
    dR3Jets_.push_back(dR->first);
    dR3JetsMass_.push_back((jets.at(dR->second.at(0)).p4()+jets.at(dR->second.at(1)).p4()+jets.at(dR->second.at(2)).p4()).mass());
  }

  std::vector< std::pair< double, unsigned short > > massDiff2W;

  for(std::vector< double >::const_iterator mass = dRMass_.begin(); mass != dRMass_.end(); ++mass){
    massDiff2W.push_back(std::make_pair(std::abs((*mass)-80.4), mass - dRMass_.begin()));
  }

  std::sort(massDiff2W.begin(), massDiff2W.end());
  
  //std::vector<std::pair< double, std::vector<unsigned short> > > massDiff;

  for(std::vector< std::pair< double, unsigned short > >::const_iterator i = massDiff2W.begin(); i != massDiff2W.end(); ++i){
    unsigned int mass1 = i->second;
    for(std::vector< std::pair< double, unsigned short > >::const_iterator j = i + 1; j != massDiff2W.end(); ++j){
      unsigned int mass2 = j->second;
      if(dRs.at(mass1).second.at(0) != dRs.at(mass2).second.at(0) && dRs.at(mass1).second.at(0) != dRs.at(mass2).second.at(1) &&
	 dRs.at(mass1).second.at(1) != dRs.at(mass2).second.at(0) && dRs.at(mass1).second.at(1) != dRs.at(mass2).second.at(1)){
	//unsigned short combA[2] = { mass1 , mass2 };
	//std::vector<unsigned short> comb(combA, combA + sizeof(combA) / sizeof(unsigned short));
	//massDiff.push_back(std::make_pair(std::abs(dRMass_.at(mass1)-dRMass_.at(mass2)), comb));
	massDiffMWCands_.push_back(std::abs(dRMass_.at(mass1)-dRMass_.at(mass2)));
      }
    }
    if(massDiffMWCands_.size() > 20) break;
  }

  //std::sort(massDiff.begin(), massDiff.end());
  /*
  for(std::vector<std::pair< double, std::vector<unsigned short> > >::const_iterator diff = massDiff.begin(); diff != massDiff.end() ; ++diff){
    std::cout << "| "   << dRMass_.at(diff->second.at(0)) << "(" << diff->second.at(0)
	      << ") - " << dRMass_.at(diff->second.at(1)) << "(" << diff->second.at(1)
	      << ") | = " << diff->first << std::endl;
  }
  std::cout << "---------------------------------------------" << std::endl;
  */

  std::sort(TCHE_BJet_Discs_  .begin(), TCHE_BJet_Discs_  .end());
  std::sort(TCHP_BJet_Discs_  .begin(), TCHP_BJet_Discs_  .end());
  std::sort(SSVHE_BJet_Discs_ .begin(), SSVHE_BJet_Discs_ .end());
  std::sort(SSVHP_BJet_Discs_ .begin(), SSVHP_BJet_Discs_ .end());
  std::sort(CSV_BJet_Discs_   .begin(), CSV_BJet_Discs_   .end());
  std::sort(CSVMVA_BJet_Discs_.begin(), CSVMVA_BJet_Discs_.end());
  std::sort(SM_BJet_Discs_    .begin(), SM_BJet_Discs_    .end());

  std::sort(etaetaMoments_.begin(), etaetaMoments_.end());
  std::sort(etaphiMoments_.begin(), etaphiMoments_.end());
  std::sort(phiphiMoments_.begin(), phiphiMoments_.end());

  std::sort(etaetaMomentsLogEt_.begin(), etaetaMomentsLogEt_.end());
  std::sort(etaphiMomentsLogEt_.begin(), etaphiMomentsLogEt_.end());
  std::sort(phiphiMomentsLogEt_.begin(), phiphiMomentsLogEt_.end());

  std::sort(etaetaMomentsMoment_.begin(), etaetaMomentsMoment_.end());
  std::sort(etaphiMomentsMoment_.begin(), etaphiMomentsMoment_.end());
  std::sort(phiphiMomentsMoment_.begin(), phiphiMomentsMoment_.end());

  std::sort(etaetaMomentsMomentLogEt_.begin(), etaetaMomentsMomentLogEt_.end());
  std::sort(etaphiMomentsMomentLogEt_.begin(), etaphiMomentsMomentLogEt_.end());
  std::sort(phiphiMomentsMomentLogEt_.begin(), phiphiMomentsMomentLogEt_.end());

  std::sort(etaetaMomentsNoB_.begin(), etaetaMomentsNoB_.end());
  std::sort(etaphiMomentsNoB_.begin(), etaphiMomentsNoB_.end());
  std::sort(phiphiMomentsNoB_.begin(), phiphiMomentsNoB_.end());

  std::sort(M3s.begin(), M3s.end());
  M3_ = ( jets.at((M3s.back().second.at(0))).p4() +  jets.at((M3s.back().second.at(1))).p4() + jets.at((M3s.back().second.at(2))).p4() ).mass();

  sqrt_s_ = totalSystem.mass();

  for(std::vector<ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > >::const_iterator jet = fourVectors.begin(); jet != fourVectors.end(); ++jet){
    ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > boostedJet = CoMBoostTotal( *jet );
    if(jet - fourVectors.begin() > 1){
      thetaStar3jet_ += (boostedJet.Theta() > M_PI/2.)? (M_PI - boostedJet.Theta()) : boostedJet.Theta();
      sinThetaStar3jet_ += sin(boostedJet.Theta());
      EtStar3jet_ += jet->Et() * pow( sin( boostedJet.Theta() ), 2);
    }
    thetaStars_.push_back((boostedJet.Theta() > M_PI/2.)? (M_PI - boostedJet.Theta()) : boostedJet.Theta());
    EtStars_.push_back( jet->Et() * pow( sin( boostedJet.Theta() ), 2) );
  }

  theta3jet_        /= (double)fourVectors.size() - 2.;
  sinTheta3jet_     /= (double)fourVectors.size() - 2.;
  thetaStar3jet_    /= (double)fourVectors.size() - 2.;
  sinThetaStar3jet_ /= (double)fourVectors.size() - 2.;
  
  EventShapeVariables eventshape(makeVecForEventShape(jets));

  aplanarity_  = eventshape.aplanarity();
  sphericity_  = eventshape.sphericity();
  circularity_ = eventshape.circularity();
  isotropy_    = eventshape.isotropy();
  C_           = eventshape.C();
  D_           = eventshape.D();

  EventShapeVariables eventshapeAll(makeVecForEventShape(jets,false));

  aplanarityAll_  = eventshapeAll.aplanarity();
  sphericityAll_  = eventshapeAll.sphericity();
  circularityAll_ = eventshapeAll.circularity();
  isotropyAll_    = eventshapeAll.isotropy();
  CAll_           = eventshapeAll.C();
  DAll_           = eventshapeAll.D();

  EventShapeVariables eventshapeAllCMS(makeVecForEventShape(jets,false,CoMBoostTotal));

  aplanarityAllCMS_  = eventshapeAllCMS.aplanarity();
  sphericityAllCMS_  = eventshapeAllCMS.sphericity();
  circularityAllCMS_ = eventshapeAllCMS.circularity();
  isotropyAllCMS_    = eventshapeAllCMS.isotropy();
  CAllCMS_           = eventshapeAllCMS.C();
  DAllCMS_           = eventshapeAllCMS.D();

  Thrust thrustAlgo(jets.begin(), jets.end());
  thrust_ = thrustAlgo.thrust();

  Thrust thrustAlgoCMS(boostedJets.begin(), boostedJets.end());
  thrustCMS_ = thrustAlgoCMS.thrust();

}

TtFullHadSignalSel::~TtFullHadSignalSel() 
{
}

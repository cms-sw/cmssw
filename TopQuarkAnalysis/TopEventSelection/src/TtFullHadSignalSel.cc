#include "TopQuarkAnalysis/TopEventSelection/interface/TtFullHadSignalSel.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "TVector3.h"

TtFullHadSignalSel::TtFullHadSignalSel():
  pt1_(-1.), pt2_(-1.), pt3_(-1.), pt4_(-1.), pt5_(-1.), pt6_(-1.)
{
}

std::vector<math::XYZVector> makeVecForEventShape(std::vector<pat::Jet> jets, double scale = 1.) {
  std::vector<math::XYZVector> p;
  unsigned int i=1;
  for (std::vector<pat::Jet>::const_iterator jet = jets.begin(); jet != jets.end(); ++jet) {
    math::XYZVector Vjet(jet->px() * scale, jet->py() * scale, jet->pz() * scale);
    p.push_back(Vjet);
    ++i;
    if(i==6) break;
  }
  return p;
}

TtFullHadSignalSel::TtFullHadSignalSel(const std::vector<pat::Jet>& jets)
{

  H_      = 0;
  Ht_     = 0;
  Ht123_  = 0;
  Ht3jet_ = 0;
  Et56_   = 0;
  sqrt_s_ = 0;
  M3_     = 0;
  
  TCHP_Bjets_ = 0;
  SSV_Bjets_  = 0;
  CSV_Bjets_  = 0;
  SM_Bjets_   = 0;

  TCHP_Bjet1_ = 0;
  TCHP_Bjet2_ = 0;
  TCHP_Bjet3_ = 0;
  TCHP_Bjet4_ = 0;
  TCHP_Bjet5_ = 0;
  TCHP_Bjet6_ = 0;
  SSV_Bjet1_  = 0; 
  SSV_Bjet2_  = 0; 
  SSV_Bjet3_  = 0; 
  SSV_Bjet4_  = 0; 
  SSV_Bjet5_  = 0; 
  SSV_Bjet6_  = 0; 
  CSV_Bjet1_  = 0; 
  CSV_Bjet2_  = 0; 
  CSV_Bjet3_  = 0; 
  CSV_Bjet4_  = 0; 
  CSV_Bjet5_  = 0; 
  CSV_Bjet6_  = 0; 
  SM_Bjet1_   = 0;  
  SM_Bjet2_   = 0;  
  SM_Bjet3_   = 0;  
  SM_Bjet4_   = 0;  
  SM_Bjet5_   = 0;  
  SM_Bjet6_   = 0;  

  jet1_etaetaMoment_ = 0;
  jet2_etaetaMoment_ = 0;
  jet3_etaetaMoment_ = 0;
  jet4_etaetaMoment_ = 0;
  jet5_etaetaMoment_ = 0;
  jet6_etaetaMoment_ = 0;
  jet1_etaphiMoment_ = 0;
  jet2_etaphiMoment_ = 0;
  jet3_etaphiMoment_ = 0;
  jet4_etaphiMoment_ = 0;
  jet5_etaphiMoment_ = 0;
  jet6_etaphiMoment_ = 0;
  jet1_phiphiMoment_ = 0;
  jet2_phiphiMoment_ = 0;
  jet3_phiphiMoment_ = 0;
  jet4_phiphiMoment_ = 0;
  jet5_phiphiMoment_ = 0;
  jet6_phiphiMoment_ = 0;
  
  aplanarity_  = 0;
  sphericity_  = 0;
  circularity_ = 0;
  isotropy_ = 0;
  C_ = 0;
  D_ = 0;

  dRMin1_        = 0;	   
  dRMin2_        = 0;	   
  sumDR3JetMin1_ = 0;	   
  sumDR3JetMin2_ = 0;	   
			   
  dRMin1Mass_        = 0;	   
  dRMin2Mass_        = 0;	   
  sumDR3JetMin1Mass_ = 0;
  sumDR3JetMin2Mass_ = 0;

  std::list< double > TCHP_Bjet_Discis;
  std::list< double > SSV_Bjet_Discis;
  std::list< double > CSV_Bjet_Discis;
  std::list< double > SM_Bjet_Discis;

  std::list< std::pair< double, std::pair< unsigned int, unsigned int > > > dRs;
  std::list< std::pair< double, std::pair< std::pair< unsigned int, unsigned int >, unsigned int > > > dRs3Jets;

  unsigned int jetCounter = 1;
  for(std::vector<pat::Jet>::const_iterator jet = jets.begin(); jet != jets.end(); ++jet, ++jetCounter){

    H_      += jet->energy();
    Ht_     += jet->et();
    Ht3jet_ += jet->et();
    
    TCHP_Bjet_Discis.push_back( jet->bDiscriminator("trackCountingHighPurBJetTags")    );
    SSV_Bjet_Discis.push_back(  jet->bDiscriminator("simpleSecondaryVertexBJetTags")   );
    CSV_Bjet_Discis.push_back(  jet->bDiscriminator("combinedSecondaryVertexBJetTags") );
    SM_Bjet_Discis.push_back(   jet->bDiscriminator("softMuonBJetTags")                );
    
    if      (jetCounter == 1) {
      pt1_ = jet->pt();
      Ht3jet_ -= jet->et();
      Ht123_  += jet->et();
      jet1_etaetaMoment_ = jet->etaetaMoment();
      jet1_etaphiMoment_ = jet->etaphiMoment();
      jet1_phiphiMoment_ = jet->phiphiMoment();
    }
    else if (jetCounter == 2) {
      pt2_ = jet->pt();
      Ht3jet_ -= jet->et();
      Ht123_  += jet->et();
      jet2_etaetaMoment_ = jet->etaetaMoment();
      jet2_etaphiMoment_ = jet->etaphiMoment();
      jet2_phiphiMoment_ = jet->phiphiMoment();
    }
    else if (jetCounter == 3) {
      pt3_ = jet->pt();
      Ht123_  += jet->et();
      jet3_etaetaMoment_ = jet->etaetaMoment();
      jet3_etaphiMoment_ = jet->etaphiMoment();
      jet3_phiphiMoment_ = jet->phiphiMoment();
    }
    else if (jetCounter == 4) {
      pt4_ = jet->pt();
      jet4_etaetaMoment_ = jet->etaetaMoment();
      jet4_etaphiMoment_ = jet->etaphiMoment();
      jet4_phiphiMoment_ = jet->phiphiMoment();
    }
    else if (jetCounter == 5) {
      pt5_ = jet->pt();
      jet5_etaetaMoment_ = jet->etaetaMoment();
      jet5_etaphiMoment_ = jet->etaphiMoment();
      jet5_phiphiMoment_ = jet->phiphiMoment();
      Et56_ += jet->et();
    }
    else if (jetCounter == 6) {
      pt6_ = jet->pt();
      jet6_etaetaMoment_ = jet->etaetaMoment();
      jet6_etaphiMoment_ = jet->etaphiMoment();
      jet6_phiphiMoment_ = jet->phiphiMoment();
      Et56_ += jet->et();
    }

    if(jet->bDiscriminator("trackCountingHighPurBJetTags")    > 2.17) ++TCHP_Bjets_;
    if(jet->bDiscriminator("simpleSecondaryVertexBJetTags")   > 2.02) ++SSV_Bjets_;
    if(jet->bDiscriminator("combinedSecondaryVertexBJetTags") > 0.9 ) ++CSV_Bjets_;
    if(jet->bDiscriminator("softMuonBJetTags")                > 0.3 ) ++SM_Bjets_;
    
    unsigned int jetCounter2 = jetCounter + 1;
    for(std::vector<pat::Jet>::const_iterator jet2 = jet; (jet2 != jets.end() && jet2 != (--jets.end()) && jet2 != (--(--jets.end()))); ++jet2, ++jetCounter2){
      ++jet2;
      dRs.push_back( std::make_pair( deltaR( jet->phi(), jet->eta(), jet2->phi(), jet2->eta() ), std::make_pair( jetCounter-1, jetCounter2-1 ) ) );
      
      unsigned int jetCounter3 = jetCounter2 + 1;
      for(std::vector<pat::Jet>::const_iterator jet3 = jet2; (jet3 != jets.end() && jet3 != (--jets.end()) && jet3 != (--(--jets.end())) && jet3 != (--(--(--jets.end())))); ++jet3, ++jetCounter3){
	++jet3;
	double dR1 = deltaR( jet->phi() , jet->eta() , jet2->phi(), jet2->eta() );
	double dR2 = deltaR( jet->phi() , jet->eta() , jet3->phi(), jet3->eta() );
	double dR3 = deltaR( jet2->phi(), jet2->eta(), jet3->phi(), jet3->eta() );
	dRs3Jets.push_back( std::make_pair( dR1+dR2+dR3, std::make_pair( std::make_pair( jetCounter-1, jetCounter2-1 ), jetCounter3-1 ) ) );
      }
    }
    
  }
  
  dRs.sort();
  dRs3Jets.sort();
  
  dRMin1_ = dRs.begin()->first;
  dRMin1Mass_ = (jets[dRs.begin()->second.first].p4()+jets[dRs.begin()->second.second].p4()).mass();
  sumDR3JetMin1_ = dRs3Jets.begin()->first;
  sumDR3JetMin1Mass_ = (jets[dRs3Jets.begin()->second.first.first].p4()+jets[dRs3Jets.begin()->second.first.second].p4()+jets[dRs3Jets.begin()->second.second].p4()).mass();
  
  for(std::list< std::pair< double, std::pair< unsigned int, unsigned int > > >::const_iterator dR = ++dRs.begin(); dR != dRs.end(); ++dR){
    if( (dR->second.first  != dRs.begin()->second.first) && (dR->second.first  != dRs.begin()->second.second) &&
	(dR->second.second != dRs.begin()->second.first) && (dR->second.second != dRs.begin()->second.second) ){
      dRMin2_ = dR->first;
      dRMin2Mass_ = (jets[dR->second.first].p4()+jets[dR->second.second].p4()).mass();
      break;
    }
  }

  for(std::list< std::pair< double, std::pair< std::pair< unsigned int, unsigned int >, unsigned int > > >::const_iterator dR = ++dRs3Jets.begin(); dR != dRs3Jets.end(); ++dR){
    if( (dR->second.first.first  != dRs3Jets.begin()->second.first.first) && (dR->second.first.first  != dRs3Jets.begin()->second.first.second) &&
	(dR->second.first.second != dRs3Jets.begin()->second.first.first) && (dR->second.first.second != dRs3Jets.begin()->second.first.second) &&
	(dR->second.first.first  != dRs3Jets.begin()->second.second)      && (dR->second.first.second != dRs3Jets.begin()->second.second)       ){
      sumDR3JetMin2_ = dR->first;
      sumDR3JetMin2Mass_ = (jets[dR->second.first.first].p4()+jets[dR->second.first.second].p4()+jets[dR->second.second].p4()).mass();
      break;
    }
  }
  
  TCHP_Bjet_Discis.sort();
  SSV_Bjet_Discis.sort();
  CSV_Bjet_Discis.sort();
  SM_Bjet_Discis.sort();
  
  unsigned int counter = 1;
  for(std::list< double >::const_reverse_iterator jet = TCHP_Bjet_Discis.rbegin(); jet != TCHP_Bjet_Discis.rend(); ++jet, ++counter){
    if     (counter == 1) TCHP_Bjet1_ = *jet;
    else if(counter == 2) TCHP_Bjet2_ = *jet;
    else if(counter == 3) TCHP_Bjet3_ = *jet;
    else if(counter == 4) TCHP_Bjet4_ = *jet;
    else if(counter == 5) TCHP_Bjet5_ = *jet;
    else if(counter == 6) TCHP_Bjet6_ = *jet;
  }

  counter = 1;
  for(std::list< double >::const_reverse_iterator jet = SSV_Bjet_Discis.rbegin(); jet != SSV_Bjet_Discis.rend(); ++jet, ++counter){
    if     (counter == 1) SSV_Bjet1_ = *jet;
    else if(counter == 2) SSV_Bjet2_ = *jet;
    else if(counter == 3) SSV_Bjet3_ = *jet;
    else if(counter == 4) SSV_Bjet4_ = *jet;
    else if(counter == 5) SSV_Bjet5_ = *jet;
    else if(counter == 6) SSV_Bjet6_ = *jet;
  }

  counter = 1;
  for(std::list< double >::const_reverse_iterator jet = CSV_Bjet_Discis.rbegin(); jet != CSV_Bjet_Discis.rend(); ++jet, ++counter){
    if     (counter == 1) CSV_Bjet1_ = *jet;
    else if(counter == 2) CSV_Bjet2_ = *jet;
    else if(counter == 3) CSV_Bjet3_ = *jet;
    else if(counter == 4) CSV_Bjet4_ = *jet;
    else if(counter == 5) CSV_Bjet5_ = *jet;
    else if(counter == 6) CSV_Bjet6_ = *jet;
  }

  counter = 1;
  for(std::list< double >::const_reverse_iterator jet = SM_Bjet_Discis.rbegin(); jet != SM_Bjet_Discis.rend(); ++jet, ++counter){
    if     (counter == 1) SM_Bjet1_ = *jet;
    else if(counter == 2) SM_Bjet2_ = *jet;
    else if(counter == 3) SM_Bjet3_ = *jet;
    else if(counter == 4) SM_Bjet4_ = *jet;
    else if(counter == 5) SM_Bjet5_ = *jet;
    else if(counter == 6) SM_Bjet6_ = *jet;
  }
  
  if(jets.size() > 2){

    M3_ = (jets[0].p4() + jets[1].p4() + jets[2].p4()).mass();

    if(jets.size() > 5){
      sqrt_s_ = (jets[0].p4() + jets[1].p4() + jets[2].p4() + jets[3].p4() + jets[4].p4() + jets[5].p4()).mass();
    }
  }

  EventShapeVariables eventshape(makeVecForEventShape(jets));

  aplanarity_  = eventshape.aplanarity();
  sphericity_  = eventshape.sphericity();
  circularity_ = eventshape.circularity();
  isotropy_    = eventshape.isotropy();
  C_           = eventshape.C();
  D_           = eventshape.D();

}

TtFullHadSignalSel::~TtFullHadSignalSel() 
{
}

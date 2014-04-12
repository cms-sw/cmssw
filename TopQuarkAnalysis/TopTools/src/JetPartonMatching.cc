#include "TopQuarkAnalysis/TopTools/interface/JetPartonMatching.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <Math/VectorUtil.h>

JetPartonMatching::JetPartonMatching(const std::vector<const reco::Candidate*>& p, const std::vector<reco::GenJet>& j,
				     const int algorithm = totalMinDist, const bool useMaxDist = true, 
				     const bool useDeltaR = true, const double maxDist = 0.3)
  : partons(p), algorithm_(algorithm), useMaxDist_(useMaxDist), useDeltaR_(useDeltaR), maxDist_(maxDist)
{
  std::vector<const reco::Candidate*> js;
  for(unsigned int i=0; i<j.size(); ++i) js.push_back( &(j[i]) );
  jets = js;
  calculate();
}

JetPartonMatching::JetPartonMatching(const std::vector<const reco::Candidate*>& p, const std::vector<reco::CaloJet>& j,
				     const int algorithm = totalMinDist, const bool useMaxDist = true,
				     const bool useDeltaR = true, const double maxDist = 0.3)
  : partons(p), algorithm_(algorithm), useMaxDist_(useMaxDist), useDeltaR_(useDeltaR), maxDist_(maxDist)
{
  std::vector<const reco::Candidate*> js;
  for(unsigned int i = 0; i < j.size(); ++i) js.push_back( &(j[i]) );
  jets = js; 
  calculate(); 
}

JetPartonMatching::JetPartonMatching(const std::vector<const reco::Candidate*>& p, const std::vector<pat::Jet>& j,
				     const int algorithm = totalMinDist, const bool useMaxDist = true,
				     const bool useDeltaR = true, const double maxDist = 0.3)
  : partons(p), algorithm_(algorithm), useMaxDist_(useMaxDist), useDeltaR_(useDeltaR), maxDist_(maxDist)
{
  std::vector<const reco::Candidate*> js;
  for(unsigned int i = 0; i < j.size(); ++i) js.push_back( &(j[i]) );
  jets = js; 
  calculate(); 
}

JetPartonMatching::JetPartonMatching(const std::vector<const reco::Candidate*>& p, const std::vector<const reco::Candidate*>& j,
				     const int algorithm = totalMinDist, const bool useMaxDist = true,
				     const bool useDeltaR = true, const double maxDist = 0.3)
  : partons(p), jets(j), algorithm_(algorithm), useMaxDist_(useMaxDist), useDeltaR_(useDeltaR), maxDist_(maxDist)
{
  calculate();
}

void 
JetPartonMatching::calculate()
{
  // use maximal distance between objects 
  // in case of unambiguousOnly algorithmm
  if(algorithm_==unambiguousOnly) useMaxDist_=true;

  // check if there are empty partons in
  // the vector, which happpens if the 
  // event is not ttbar or the decay is 
  // not as expected
  bool emptyParton=false;
  for(unsigned int ip=0; ip<partons.size(); ++ip){
    if( partons[ip]->pdgId() ==0 ){
      emptyParton=true;
      break;
    }
  }

  // switch algorithm, default is to match
  // on the minimal sum of the distance 
  // (if jets or a parton is empty fill match with blanks)
  if( jets.empty() || emptyParton ) {
    MatchingCollection dummyMatch;
    for(unsigned int ip=0; ip<partons.size(); ++ip)
      dummyMatch.push_back( std::make_pair(ip, -1) );
    matching.push_back( dummyMatch );
  }
  else {
    switch(algorithm_) {
      
    case totalMinDist: 
      matchingTotalMinDist();    
      break;
      
    case minSumDist: 
      matchingMinSumDist();
      break;
      
    case ptOrderedMinDist: 
      matchingPtOrderedMinDist();
      break;
      
    case unambiguousOnly:
      matchingUnambiguousOnly();
      break;
      
    default:
      matchingMinSumDist();
    }
  }

  numberOfUnmatchedPartons.clear();
  sumDeltaE .clear();
  sumDeltaPt.clear();
  sumDeltaR .clear();
  for(unsigned int comb = 0; comb < matching.size(); ++comb) {
    MatchingCollection match = matching[comb];
    std::sort(match.begin(), match.end());
    matching[comb] = match;
    
    int nUnmatchedPartons = partons.size();
    for(unsigned int part=0; part<partons.size(); ++part)
      if(getMatchForParton(part,comb)>=0) --nUnmatchedPartons;

    double sumDE  = -999.;
    double sumDPt = -999.;
    double sumDR  = -999.;
    if(nUnmatchedPartons==0){
      sumDE  = 0;
      sumDPt = 0;
      sumDR  = 0;
      for(unsigned int i=0; i<match.size(); ++i){
	sumDE  += fabs(partons[match[i].first]->energy() - jets[match[i].second]->energy());
	sumDPt += fabs(partons[match[i].first]->pt()     - jets[match[i].second]->pt());
	sumDR  += distance(partons[match[i].first]->p4(), jets[match[i].second]->p4());
      }
    }

    numberOfUnmatchedPartons.push_back( nUnmatchedPartons );
    sumDeltaE .push_back( sumDE  );
    sumDeltaPt.push_back( sumDPt );
    sumDeltaR .push_back( sumDR  );
  }
}

double 
JetPartonMatching::distance(const math::XYZTLorentzVector& v1, const math::XYZTLorentzVector& v2)
{
  // calculate the distance between two lorentz vectors 
  // using DeltaR(eta, phi) or normal space angle(theta, phi)
  if(useDeltaR_) return ROOT::Math::VectorUtil::DeltaR(v1, v2);
  return ROOT::Math::VectorUtil::Angle(v1, v2);
}

void 
JetPartonMatching::matchingTotalMinDist()
{
  // match parton to jet with shortest distance
  // starting with the shortest distance available
  // apply some outlier rejection if desired

  // prepare vector of pairs with distances between
  // all partons to all jets in the input vectors
  std::vector< std::pair<double, unsigned int> > distances;
  for(unsigned int ip=0; ip<partons.size(); ++ip){
    for(unsigned int ij=0; ij<jets.size(); ++ij){ 
      double dist = distance(jets[ij]->p4(), partons[ip]->p4());
      distances.push_back(std::pair<double, unsigned int>(dist, ip*jets.size()+ij));
    }
  }
  std::sort(distances.begin(), distances.end());

  MatchingCollection match;

  while(match.size() < partons.size()){
    unsigned int partonIndex = distances[0].second/jets.size();
    int jetIndex = distances[0].second-jets.size()*partonIndex;
    
    // use primitive outlier rejection if desired
    if(useMaxDist_&& distances[0].first>maxDist_) jetIndex = -1;

    // prevent underflow in case of too few jets
    if( distances.empty() )
      match.push_back(std::make_pair(partonIndex, -1));
    else
      match.push_back(std::make_pair(partonIndex, jetIndex));
    
    // remove all values for the matched parton 
    // and the matched jet
    for(unsigned int a=0; a<distances.size(); ++a){
      unsigned int pIndex = distances[a].second/jets.size();
      int jIndex = distances[a].second-jets.size()*pIndex;
      if((pIndex == partonIndex) || (jIndex == jetIndex)){
	distances.erase(distances.begin()+a, distances.begin()+a+1); 
	--a;
      }
    }
  }

  matching.clear();
  matching.push_back( match );
  return;
}

void 
JetPartonMatching::minSumDist_recursion(const unsigned int ip, std::vector<unsigned int> & jetIndices,
					std::vector<bool> & usedJets, std::vector<std::pair<double, MatchingCollection> > & distMatchVec)
{
  // build up jet combinations recursively
  if(ip<partons.size()){
    for(unsigned int ij=0; ij<jets.size(); ++ij){
      if(usedJets[ij]) continue;
      usedJets[ij] = true;
      jetIndices[ip] = ij;
      minSumDist_recursion(ip+1, jetIndices, usedJets, distMatchVec);
      usedJets[ij] = false;
    }
    return;
  }

  // calculate sumDist for each completed combination
  double sumDist = 0;
  MatchingCollection match;
  for(unsigned int ip=0; ip<partons.size(); ++ip){
    double dist  = distance(partons[ip]->p4(), jets[jetIndices[ip]]->p4());
    if(useMaxDist_ && dist > maxDist_) return; // outlier rejection
    sumDist += distance(partons[ip]->p4(), jets[jetIndices[ip]]->p4());
    match.push_back(std::make_pair(ip, jetIndices[ip]));
  }

  distMatchVec.push_back( std::make_pair(sumDist, match)  );
  return;
}

void JetPartonMatching::matchingMinSumDist()
{
  // match partons to jets with minimal sum of
  // the distances between all partons and jets

  std::vector<std::pair<double, MatchingCollection> > distMatchVec;

  std::vector<bool> usedJets;
  for(unsigned int i=0; i<jets.size(); ++i){
    usedJets.push_back(false);
  }

  std::vector<unsigned int> jetIndices;
  jetIndices.reserve(partons.size());

  minSumDist_recursion(0, jetIndices, usedJets, distMatchVec);

  std::sort(distMatchVec.begin(), distMatchVec.end());

  matching.clear();

  if(distMatchVec.empty()) {
    MatchingCollection dummyMatch;
    for(unsigned int ip=0; ip<partons.size(); ++ip)
      dummyMatch.push_back(std::make_pair(ip, -1));
    matching.push_back( dummyMatch );
  }
  else
    for(unsigned int i=0; i<distMatchVec.size(); ++i)
      matching.push_back( distMatchVec[i].second );

  return;
}

void 
JetPartonMatching::matchingPtOrderedMinDist()
{
  // match partons to jets with minimal sum of
  // the distances between all partons and jets
  // order partons in pt first
  std::vector<std::pair <double, unsigned int> > ptOrderedPartons;

  for(unsigned int ip=0; ip<partons.size(); ++ip)
    ptOrderedPartons.push_back(std::make_pair(partons[ip]->pt(), ip));

  std::sort(ptOrderedPartons.begin(), ptOrderedPartons.end());
  std::reverse(ptOrderedPartons.begin(), ptOrderedPartons.end());

  std::vector<unsigned int> jetIndices;
  for(unsigned int ij=0; ij<jets.size(); ++ij) jetIndices.push_back(ij);

  MatchingCollection match;

  for(unsigned int ip=0; ip<ptOrderedPartons.size(); ++ip){
    double minDist = 999.;
    int ijMin = -1;

    for(unsigned int ij=0; ij<jetIndices.size(); ++ij){
      double dist = distance(partons[ptOrderedPartons[ip].second]->p4(), jets[jetIndices[ij]]->p4());
      if(dist < minDist){
	if(!useMaxDist_ || dist <= maxDist_){
	  minDist = dist;
	  ijMin = ij;
	}
      }
    }
    
    if(ijMin >= 0){
      match.push_back( std::make_pair(ptOrderedPartons[ip].second, jetIndices[ijMin]) );
      jetIndices.erase(jetIndices.begin() + ijMin, jetIndices.begin() + ijMin + 1);
    }
    else
      match.push_back( std::make_pair(ptOrderedPartons[ip].second, -1) );
  }

  matching.clear();
  matching.push_back( match );
  return;
}

void 
JetPartonMatching::matchingUnambiguousOnly()
{
  // match partons to jets, only accept event 
  // if there are no ambiguities
  std::vector<bool> jetMatched;
  for(unsigned int ij=0; ij<jets.size(); ++ij) jetMatched.push_back(false);

  MatchingCollection match;
  
  for(unsigned int ip=0; ip<partons.size(); ++ip){
    int iMatch = -1;
    for(unsigned int ij=0; ij<jets.size(); ++ij){
      double dist = distance(partons[ip]->p4(), jets[ij]->p4());
      if(dist <= maxDist_){
	if(!jetMatched[ij]){ // new match for jet
	  jetMatched[ij] = true;
	  if(iMatch == -1) // new match for parton and jet
	    iMatch = ij;
	  else // parton already matched: ambiguity!
	    iMatch = -2;
	}
	else // jet already matched: ambiguity!
	  iMatch = -2;
      }
    }
    match.push_back(std::make_pair(ip, iMatch));
  }

  matching.clear();
  matching.push_back( match );
  return;
}

int
JetPartonMatching::getMatchForParton(const unsigned int part, const unsigned int comb)
{
  // return index of the matched jet for a given parton
  // (if arguments for parton index and combinatoric index
  // are in the valid range)
  if(comb >= matching.size()) return -9;
  if(part >= matching[comb].size()) return -9;
  return (matching[comb])[part].second;
}

std::vector<int>
JetPartonMatching::getMatchesForPartons(const unsigned int comb)
{
  // return a vector with the indices of the matched jets
  // (ordered according to the vector of partons)
  std::vector<int> jetIndices;
  for(unsigned int part=0; part<partons.size(); ++part)
    jetIndices.push_back( getMatchForParton(part, comb) );
  return jetIndices;
}

double 
JetPartonMatching::getDistanceForParton(const unsigned int part, const unsigned int comb)
{
  // get the distance between parton and its best matched jet
  if(getMatchForParton(part, comb) < 0) return -999.;
  return distance( jets[getMatchForParton(part,comb)]->p4(), partons[part]->p4() );
}

double 	
JetPartonMatching::getSumDistances(const unsigned int comb)
{
  // get sum of distances between partons and matched jets
  double sumDists = 0.;
  for(unsigned int part=0; part<partons.size(); ++part){
    double dist = getDistanceForParton(part, comb);
    if(dist < 0.) return -999.;
    sumDists += dist;
  }
  return sumDists;
}

void
JetPartonMatching::print()
{
  //report using MessageLogger
  edm::LogInfo log("JetPartonMatching");
  log << "++++++++++++++++++++++++++++++++++++++++++++++ \n";
  log << " algorithm : ";
  switch(algorithm_) {
  case totalMinDist     : log << "totalMinDist    "; break;
  case minSumDist       : log << "minSumDist      "; break;
  case ptOrderedMinDist : log << "ptOrderedMinDist"; break;
  case unambiguousOnly  : log << "unambiguousOnly "; break;
  default               : log << "UNKNOWN         ";
  }
  log << "\n";
  log << " useDeltaR : ";
  switch(useDeltaR_) {
  case false : log << "false"; break;
  case true  : log << "true ";
  }
  log << "\n";
  log << " useMaxDist: ";
  switch(useMaxDist_) {
  case false : log << "false"; break;
  case true  : log << "true ";
  }
  log << "      maxDist: " << maxDist_ << "\n";
  log << " number of partons / jets: " << partons.size() << " / " << jets.size() << "\n";
  log << " number of available combinations: " << getNumberOfAvailableCombinations() << "\n";
  for(unsigned int comb = 0; comb < matching.size(); ++comb) {
    log << " -------------------------------------------- \n";
    log << " ind. of matched jets:";
    for(unsigned int part = 0; part < partons.size(); ++part)
      log << std::setw(4) << getMatchForParton(part, comb);
    log << "\n";
    log << " sumDeltaR             : " << getSumDeltaR(comb) << "\n";
    log << " sumDeltaPt / sumDeltaE: " << getSumDeltaPt(comb) << " / "  << getSumDeltaE(comb);
    log << "\n";
  }
  log << "++++++++++++++++++++++++++++++++++++++++++++++";
}

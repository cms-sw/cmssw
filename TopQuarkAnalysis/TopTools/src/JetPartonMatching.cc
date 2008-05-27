#include "TopQuarkAnalysis/TopTools/interface/JetPartonMatching.h"

JetPartonMatching::JetPartonMatching(const std::vector<const reco::Candidate*>& p, const std::vector<reco::GenJet>& j,
				     const int algorithm = 3, const bool useMaxDist = true, 
				     const bool useDeltaR = true, const double maxDist = 0.3)
  : partons(p), algorithm_(algorithm), useMaxDist_(useMaxDist), useDeltaR_(useDeltaR), maxDist_(maxDist)
{
  std::vector<const reco::Candidate*> js;
  for(unsigned int i=0; i<j.size(); ++i) js.push_back( &(j[i]) );
  jets = js;
  calculate();
}

JetPartonMatching::JetPartonMatching(const std::vector<const reco::Candidate*>& p, const std::vector<reco::CaloJet>& j,
				     const int algorithm = 3, const bool useMaxDist = true,
				     const bool useDeltaR = true, const double maxDist = 0.3)
  : partons(p), algorithm_(algorithm), useMaxDist_(useMaxDist), useDeltaR_(useDeltaR), maxDist_(maxDist)
{
  std::vector<const reco::Candidate*> js;
  for(unsigned int i = 0; i < j.size(); ++i) js.push_back( &(j[i]) );
  jets = js; 
  calculate(); 
}

JetPartonMatching::JetPartonMatching(const std::vector<const reco::Candidate*>& p, const std::vector<const reco::Candidate*>& j,
				     const int algorithm = 3, const bool useMaxDist = true,
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
  
  // switch algorithm, default is to match
  // on the minimal sum of the distance 
  // if jets is empty fill match with blanks
  if( jets.empty() )
    for(unsigned int ip=0; ip<partons.size(); ++ip)
      matching.push_back(std::make_pair(ip, -1));
  else{
    switch(algorithm_){
      
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
  std::sort(matching.begin(), matching.end());
  
  numberOfUnmatchedPartons = partons.size();
  for(unsigned int ip=0; ip<partons.size(); ++ip)
    if(getMatchForParton(ip)>=0) --numberOfUnmatchedPartons;
  
  if(numberOfUnmatchedPartons>0){
    sumDeltaE = -999;
    sumDeltaPt= -999;
    sumDeltaR = -999;
  }
  else{
    sumDeltaE = 0;
    sumDeltaPt= 0;
    sumDeltaR = 0;
    for(size_t i=0; i<matching.size(); ++i){
      sumDeltaE += fabs(partons[matching[i].first]->energy() - jets[matching[i].second]->energy());
      sumDeltaPt+= fabs(partons[matching[i].first]->pt() - jets[matching[i].second]->pt());
      sumDeltaR += distance(partons[matching[i].first]->p4(), jets[matching[i].second]->p4());
    }
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

  while(matching.size() < partons.size()){
    unsigned int partonIndex = distances[0].second/jets.size();
    int jetIndex = distances[0].second-jets.size()*partonIndex;
    
    // use primitive outlier rejection if desired
    if(useMaxDist_&& distances[0].first>maxDist_) jetIndex = -1;
    
    matching.push_back(std::make_pair(partonIndex, jetIndex));
    
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
  return;
}

void 
JetPartonMatching::minSumDist_recursion(const unsigned int ip, std::vector<unsigned int> & jetIndices,
					std::vector<bool> & usedJets, std::vector<int> & ijMin, double & minSumDist)
{
  // build up jet combinations recursively
  if(ip<partons.size()){
    for(unsigned int ij=0; ij<jets.size(); ++ij){
      if(usedJets[ij]) continue;
      usedJets[ij] = true;
      jetIndices[ip] = ij;
      minSumDist_recursion(ip+1, jetIndices, usedJets, ijMin, minSumDist);
      usedJets[ij] = false;
    }
    return;
  }

  // calculate sumDist for each completed combination
  double sumDist = 0;
  for(unsigned int ip=0; ip<partons.size(); ++ip){
    double dist  = distance(partons[ip]->p4(), jets[jetIndices[ip]]->p4());
    if(useMaxDist_ && dist > maxDist_) return; // outlier rejection
    sumDist += distance(partons[ip]->p4(), jets[jetIndices[ip]]->p4());  
  }

  if(sumDist<minSumDist){
    minSumDist = sumDist;
    for(unsigned int ip = 0; ip < partons.size(); ip++) ijMin[ip] = jetIndices[ip];
  }
  return;
}

void JetPartonMatching::matchingMinSumDist()
{
  // match partons to jets with minimal sum of
  // the distances between all partons and jets
  double minSumDist = 999.;
  std::vector<int> ijMin;
  
  for(unsigned int ip=0; ip<partons.size(); ++ip)
    ijMin.push_back(-1);

  std::vector<bool> usedJets;
  for(unsigned int i=0; i<jets.size(); ++i){
    usedJets.push_back(false);
  }

  std::vector<unsigned int> jetIndices;
  jetIndices.reserve(partons.size());

  minSumDist_recursion(0, jetIndices, usedJets, ijMin, minSumDist);

  for(unsigned int ip=0; ip<partons.size(); ++ip)
    matching.push_back(std::make_pair(ip, ijMin[ip]));
  
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
      matching.push_back(std::make_pair(ptOrderedPartons[ip].second, jetIndices[ijMin]));
      jetIndices.erase(jetIndices.begin() + ijMin, jetIndices.begin() + ijMin + 1);
    }
    else
      matching.push_back(std::make_pair(ptOrderedPartons[ip].second, -1));
  }
  return;
}

void 
JetPartonMatching::matchingUnambiguousOnly()
{
  // match partons to jets, only accept event 
  // if there are no ambiguouities
  std::vector<bool> jetMatched;
  for(unsigned int ij=0; ij<jets.size(); ++ij) jetMatched.push_back(false);
  
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
    matching.push_back(std::make_pair(ip, iMatch));
  }
  return;
}

double 
JetPartonMatching::getAngleForParton(unsigned int ip)
{
  // get the angle between parton ip and its best 
  // matched jet (kept for backwards compatibility)
  if(getMatchForParton(ip) > -1) 
    return distance( jets[getMatchForParton(ip)]->p4(), partons[ip]->p4() );
  return -999;
}

double 	
JetPartonMatching::getSumAngles()
{
  // get sum of the angles between partons and 
  // matched jets (kept for backwards compatibility)
  double sumAngles = 0;
  for(size_t i=0; i<matching.size(); ++i){
    sumAngles += getAngleForParton(i);
  }
  return sumAngles;
}

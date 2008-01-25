
//
// Author:  Jan Heyninck
// Created: Tue Apr  3 17:33:23 PDT 2007
//
// $Id: JetPartonMatching.h,v 1.1 2007/07/04 16:51:37 heyninck Exp $
//

#ifndef JetPartonMatching_h
#define JetPartonMatching_h

/**
  \class    JetPartonMatching JetPartonMatching.h "TopQuarkAnalysis/TopTools/interface/JetPartonMatching.h"
  \brief    Help functionalities to implement and evaluate LR ratio method

  \author   Jan Heyninck
  \version  $Id: JetPartonMatching.h,v 1.1 2007/07/04 16:51:37 heyninck Exp $
*/

#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include <vector>
#include <iostream>
#include <Math/VectorUtil.h>

using namespace std;

class JetPartonMatching {

  public:
    JetPartonMatching();
    JetPartonMatching(vector<const reco::GenParticle*> &, vector<reco::GenJet> &, int);
    JetPartonMatching(vector<const reco::GenParticle*> &, vector<reco::CaloJet> &, int);
    JetPartonMatching(vector<const reco::GenParticle*> &, vector<const reco::Candidate*> &, int);
    ~JetPartonMatching();	

    vector<pair<unsigned int,unsigned int> > getMatching() { return matching; }
    int    getMatchForParton(unsigned int);
    double getAngleForParton(unsigned int);
    double getSumAngles();
     
   
  private:
    vector<const reco::GenParticle*> partons; 
    vector<const reco::Candidate*> jets; 
    int spaceAngleOrDeltaR;
    void calculate();
    vector<pair <unsigned int,unsigned int> > matching;
};

#endif

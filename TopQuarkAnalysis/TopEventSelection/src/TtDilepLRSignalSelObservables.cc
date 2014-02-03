#include "TopQuarkAnalysis/TopEventSelection/interface/TtDilepLRSignalSelObservables.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Common/interface/Handle.h"
#include "AnalysisDataFormats/TopObjects/interface/TtGenEvent.h"

/************** Definition of the functions of the class ***************/

//Constructor
TtDilepLRSignalSelObservables::TtDilepLRSignalSelObservables(){
count1=0; count2=0; count3=0;
count4=0; count5=0; count3=0;
}

// Destructor
TtDilepLRSignalSelObservables::~TtDilepLRSignalSelObservables(){
}

std::vector< TtDilepLRSignalSelObservables::IntBoolPair >
TtDilepLRSignalSelObservables::operator() (TtDilepEvtSolution &solution,
	const edm::Event & iEvent, bool matchOnly)
{
  evtselectVarVal.clear();
  evtselectVarMatch.clear();

  // Check whether the objects are matched:
  bool matchB1 = false;
  bool matchB2 = false;
  bool matchB = false;
  bool matchBbar = false;
  bool matchLeptPos =  false;
  bool matchLeptNeg =  false;

  edm::Handle<TtGenEvent> genEvent;
  iEvent.getByLabel ("genEvt",genEvent);

  if (!genEvent.failedToGet() && genEvent.isValid()) {
    // std::cout <<std::endl;
    double dr, dr1, dr2;

    if (genEvent->isFullLeptonic()) {
      // Match the leptons, by type and deltaR
      dr = DeltaR<reco::Particle, reco::GenParticle>()(solution.getLeptPos(), *(solution.getGenLepp()));
      matchLeptPos = (
	( ((solution.getWpDecay()=="electron")&&(std::abs(solution.getGenLepp()->pdgId())==11))
       || ((solution.getWpDecay()=="muon")&&(std::abs(solution.getGenLepp()->pdgId())==13)) )
       && (dr < 0.1) );

      dr = DeltaR<reco::Particle, reco::GenParticle>()(solution.getLeptNeg(), *(solution.getGenLepm()));
      matchLeptNeg = (
	( ((solution.getWmDecay()=="electron")&&(std::abs(solution.getGenLepm()->pdgId())==11))
           || ((solution.getWmDecay()=="muon")&&(std::abs(solution.getGenLepm()->pdgId())==13)) )
	&& (dr < 0.1) );
    }

    if (genEvent->isSemiLeptonic()) {
      int id = genEvent->singleLepton()->pdgId();

      if (id>0) {
	dr = DeltaR<reco::Particle, reco::GenParticle>()(solution.getLeptNeg(), *(genEvent->singleLepton()));
	matchLeptNeg = (
	  ( ((solution.getWmDecay()=="electron") && (id==11))
             || ((solution.getWmDecay()=="muon") && (id==13)) )
	  && (dr < 0.1) );
      } else {
	dr = DeltaR<reco::Particle, reco::GenParticle>()(solution.getLeptPos(), *(genEvent->singleLepton()));
	matchLeptPos = (
	  ( ((solution.getWpDecay()=="electron")&& (id==-11))
	 || ((solution.getWpDecay()=="muon")    && (id==-13)) )
	 && (dr < 0.1) );
      }
    }

    if (genEvent->isTtBar() && genEvent->numberOfBQuarks()>1) {
      if (solution.getJetB().partonFlavour()==5) ++count1;
      if (solution.getJetBbar().partonFlavour()==5) ++count1;

      dr1 = DeltaR<pat::Jet, reco::GenParticle>()(solution.getCalJetB(), *(genEvent->b()));
      dr2 = DeltaR<pat::Jet, reco::GenParticle>()(solution.getCalJetB(), *(genEvent->bBar()));

      matchB1= ( (dr1<0.4) || (dr2<0.4));
      matchB = ( (solution.getJetB().partonFlavour()==5) && (dr1<0.4) );
      if (matchB) ++count3;
      matchB = ( (dr1<0.4) );
      if (dr1<0.5) ++count2;
      if (dr1<0.4) ++count4;
      if (dr1<0.3) ++count5;

      dr1 = DeltaR<pat::Jet, reco::GenParticle>()(solution.getCalJetBbar(), *(genEvent->b()));
      dr2 = DeltaR<pat::Jet, reco::GenParticle>()(solution.getCalJetBbar(), *(genEvent->bBar()));

      matchBbar = ( (solution.getJetBbar().partonFlavour()==5) && (dr2<0.4) );
      if (matchBbar) ++count3;
      matchBbar = ( (dr2<0.4) );
      matchB2 = ( (dr1<0.4) || (dr2<0.4));
      if (dr2<0.5) ++count2;
      if (dr2<0.4) ++count4;
      if (dr2<0.3) ++count5;
    }

  }

  edm::Handle<std::vector<pat::Jet> > jets;
  iEvent.getByLabel(jetSource_, jets);

  //  Lower / Higher of both jet angles

  double v1 = std::abs( solution.getJetB().p4().theta() - M_PI/2 );
  double v2 = std::abs( solution.getJetBbar().p4().theta() - M_PI/2 ) ;
  fillMinMax(v1, v2, 1, evtselectVarVal, matchB1, matchB2, evtselectVarMatch);

  //  Lower / Higher of both jet pT

  double pt1 = solution.getJetB().p4().pt();
  double pt2 = solution.getJetBbar().p4().pt();
  fillMinMax(pt1, pt2, 3, evtselectVarVal, matchB1, matchB2, evtselectVarMatch);

  //  Lower / Higher of both lepton pT

  pt1 = solution.getLeptPos().p4().pt();
  pt2 = solution.getLeptNeg().p4().pt();
  fillMinMax(pt1, pt2, 5, evtselectVarVal, matchLeptPos, matchLeptNeg, evtselectVarMatch);

  // delta theta btw the b-jets

  double deltaPhi = std::abs( delta(solution.getJetB().p4().phi(),
				    solution.getJetBbar().p4().phi()) );

  evtselectVarVal.push_back(IntDblPair(7, deltaPhi));
  evtselectVarMatch.push_back(IntBoolPair(7, matchB1&&matchB2));

  // delta phi btw the b-jets

  double deltaTheta = std::abs( delta (solution.getJetBbar().p4().theta(),
				       solution.getJetB().p4().theta() ) );

  evtselectVarVal.push_back(IntDblPair(8, deltaTheta));
  evtselectVarMatch.push_back(IntBoolPair(8, matchB1&&matchB2));

  //  Lower / Higher of phi difference between the b and associated lepton

  double deltaPhi1 = std::abs ( delta( solution.getJetB().p4().phi(),
				       solution.getLeptPos().p4().phi() ) );
  double deltaPhi2 = std::abs ( delta( solution.getJetBbar().p4().phi(),
				       solution.getLeptNeg().p4().phi() ) );

  fillMinMax(deltaPhi1, deltaPhi2, 9, evtselectVarVal,
	matchB&&matchLeptPos, matchBbar&&matchLeptNeg, evtselectVarMatch);

  //  Lower / Higher of theta difference between the b and associated lepton

  double deltaTheta1 = std::abs( solution.getJetB().p4().theta() -
				 solution.getLeptPos().p4().theta() );
  double deltaTheta2 = std::abs( solution.getJetBbar().p4().theta() -
				 solution.getLeptNeg().p4().theta() );
  fillMinMax(deltaTheta1, deltaTheta2, 11, evtselectVarVal,
	matchB&&matchLeptPos, matchBbar&&matchLeptNeg, evtselectVarMatch);

  // Invariant Mass of lepton pair

  math::XYZTLorentzVector pp = solution.getLeptPos().p4() + solution.getLeptNeg().p4();
  double mass = pp.mass();
  evtselectVarVal.push_back(IntDblPair(13, mass));
  evtselectVarMatch.push_back(IntBoolPair(13, matchLeptNeg&&matchLeptPos));

  evtselectVarVal.push_back(IntDblPair(13, mass));
  evtselectVarMatch.push_back(IntBoolPair(13, matchLeptNeg&&matchLeptPos));

  std::vector <pat::Jet> jet3;
  for (int i=0;i<(int)jets->size();++i) {
if  ( ((*jets)[i].et()<solution.getJetB().et()) && ((*jets)[i].et()<solution.getJetBbar().et())) {jet3.push_back((*jets)[i]);
}}
  double jet1Ratio = 0., jet2Ratio = 0.;
  if (jet3.size()>0) {
    jet1Ratio = jet3[0].et()/solution.getJetB().et();
    jet2Ratio = jet3[0].et()/solution.getJetBbar().et();
  }
  fillMinMax(jet1Ratio, jet2Ratio, 14, evtselectVarVal,
	matchB1, matchB2, evtselectVarMatch);

  evtselectVarVal.push_back(IntDblPair(16, jets->size()));
  evtselectVarMatch.push_back(IntBoolPair(16, matchB&&matchBbar));


  if (!matchOnly) solution.setLRSignalEvtObservables(evtselectVarVal);
  return evtselectVarMatch;
}

void TtDilepLRSignalSelObservables::fillMinMax
	(double v1, double v2, int obsNbr, std::vector< IntDblPair > & varList,
	 bool match1, bool match2, std::vector< IntBoolPair > & matchList)
{
  if (v1<v2) {
    varList.push_back(IntDblPair(obsNbr, v1));
    varList.push_back(IntDblPair(obsNbr+1, v2));
    matchList.push_back(IntBoolPair(obsNbr, match1));
    matchList.push_back(IntBoolPair(obsNbr+1, match2));

  } else {
    varList.push_back(IntDblPair(obsNbr, v2));
    varList.push_back(IntDblPair(obsNbr+1, v1));
    matchList.push_back(IntBoolPair(obsNbr, match2));
    matchList.push_back(IntBoolPair(obsNbr+1, match1));
  }
}

double TtDilepLRSignalSelObservables::delta(double phi1, double phi2)
{
  double deltaPhi = phi1 - phi2;
  while (deltaPhi > M_PI) deltaPhi -= 2*M_PI;
  while (deltaPhi <= -M_PI) deltaPhi += 2*M_PI;
  return deltaPhi;
}

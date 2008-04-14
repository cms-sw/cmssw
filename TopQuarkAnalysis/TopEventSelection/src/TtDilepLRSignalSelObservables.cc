#include "TopQuarkAnalysis/TopEventSelection/interface/TtDilepLRSignalSelObservables.h"
#include "PhysicsTools/Utilities/interface/DeltaR.h"
// #include "PhysicsTools/Utilities/interface/DeltaPhi.h"
// #include "PhysicsTools/Utilities/interface/DeltaTheta.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Common/interface/Handle.h"
#include "AnalysisDataFormats/TopObjects/interface/TtGenEvent.h"

// // ROOT classes
// #include "TLorentzVector.h"
// #include "TVector.h"
// #include "TVectorD.h"
// 
// #include "TMatrix.h"
// #include "TMatrixDSymEigen.h"
// #include "TMatrixDSym.h"
// #include "TMatrixTSym.h"

using namespace reco;
using namespace std;
using namespace math;
using namespace edm;


/************** Definition of the functions of the class ***************/

//Constructor
TtDilepLRSignalSelObservables::TtDilepLRSignalSelObservables(){
count1=0; count2=0; count3=0;
count4=0; count5=0; count3=0;
}
// Destructor
TtDilepLRSignalSelObservables::~TtDilepLRSignalSelObservables(){
// cout << "Jet flavour match: " <<   count1<<" "<<  count2<<" "<< count3
// <<" "<<  count4<<" "<< count5<<endl; 
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

  try {
    // cout <<endl;
    double dr, dr1, dr2;

    Handle<TtGenEvent> genEvent;
    iEvent.getByLabel ("genEvt",genEvent);

    if (genEvent->isFullLeptonic()) {
      //cout << "Dilepton:\n";
      // Match the leptons, by type and deltaR
      dr = DeltaR<reco::Particle>()(solution.getLeptPos(), *(solution.getGenLepp()));
      matchLeptPos = (
	( ((solution.getWpDecay()=="electron")&&(abs(solution.getGenLepp()->pdgId())==11))
       || ((solution.getWpDecay()=="muon")&&(abs(solution.getGenLepp()->pdgId())==13)) )
       && (dr < 0.1) );
      // cout << solution.getWpDecay() << solution.getGenLepp()->pdgId()<<" "<<dr<<endl;

      dr = DeltaR<reco::Particle>()(solution.getLeptNeg(), *(solution.getGenLepm()));
      matchLeptNeg = (
	( ((solution.getWmDecay()=="electron")&&(abs(solution.getGenLepm()->pdgId())==11))
           || ((solution.getWmDecay()=="muon")&&(abs(solution.getGenLepm()->pdgId())==13)) )
	&& (dr < 0.1) );
      // cout << solution.getWmDecay() << solution.getGenLepm()->pdgId()<<" "<<dr<<endl;
    }

    if (genEvent->isSemiLeptonic()) {
      int id = genEvent->singleLepton()->pdgId();
      //cout << "Semi-Leptonic: ";

      if (id>0) {
	dr = DeltaR<reco::Particle>()(solution.getLeptNeg(), *(genEvent->singleLepton()));
	matchLeptNeg = (
	  ( ((solution.getWmDecay()=="electron") && (id==11))
             || ((solution.getWmDecay()=="muon") && (id==13)) )
	  && (dr < 0.1) );
	// cout << solution.getWmDecay() << id<<" "<<dr<<endl;
      } else {
	dr = DeltaR<reco::Particle>()(solution.getLeptPos(), *(genEvent->singleLepton()));
	matchLeptPos = (
	  ( ((solution.getWpDecay()=="electron")&& (id==-11))
	 || ((solution.getWpDecay()=="muon")    && (id==-13)) )
	 && (dr < 0.1) );
	// cout << solution.getWpDecay() << id<<" "<<dr<<endl;
      }
    }

    if (genEvent->isFullHadronic()) {
      // cout << "Hadronic\n";
    }
    
    if (genEvent->isTtBar() && genEvent->numberOfBQuarks()>1) {
      if (solution.getJetB().partonFlavour()==5) ++count1;
      if (solution.getJetBbar().partonFlavour()==5) ++count1;

      dr1 = DeltaR<reco::Particle>()(solution.getCalJetB(), *(genEvent->b()));
      dr2 = DeltaR<reco::Particle>()(solution.getCalJetB(), *(genEvent->bBar()));

      matchB1= ( (dr1<0.4) || (dr2<0.4));
      matchB = ( (solution.getJetB().partonFlavour()==5) && (dr1<0.4) );
      if (matchB) ++count3;
      matchB = ( (dr1<0.4) );
      if (dr1<0.5) ++count2;
      if (dr1<0.4) ++count4;
      if (dr1<0.3) ++count5;
      //cout << solution.getJetB().partonFlavour() << " "<<dr<<endl;

      dr1 = DeltaR<reco::Particle>()(solution.getCalJetBbar(), *(genEvent->b()));
      dr2 = DeltaR<reco::Particle>()(solution.getCalJetBbar(), *(genEvent->bBar()));

      matchBbar = ( (solution.getJetBbar().partonFlavour()==5) && (dr2<0.4) );
      if (matchBbar) ++count3;
      matchBbar = ( (dr2<0.4) );
      matchB2 = ( (dr1<0.4) || (dr2<0.4));
      if (dr2<0.5) ++count2;
      if (dr2<0.4) ++count4;
      if (dr2<0.3) ++count5;
      //cout << solution.getJetBbar().partonFlavour() << " "<<dr<<endl;
    }

    //Look at the b-jets:
    //cout << "Final Match: "<<   matchB<<matchBbar <<matchLeptPos <<matchLeptNeg<<endl;
    
  } catch (...){cout << "Exception\n";}
  
  Handle<vector<pat::Jet> > jets;
  iEvent.getByLabel(jetSource_, jets);
  
  //  Lower / Higher of both jet angles
  
  double v1 = abs( solution.getJetB().p4().theta() - M_PI/2 );
  double v2 = abs( solution.getJetBbar().p4().theta() - M_PI/2 ) ;
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

  double deltaPhi = abs ( delta(solution.getJetB().p4().phi(), 
			solution.getJetBbar().p4().phi()) );
//   double deltaPhi = abs( solution.getJetB().p4().DeltaPhi(
//  		solution.getJetBbar().p4() ) );


  evtselectVarVal.push_back(IntDblPair(7, deltaPhi));
  evtselectVarMatch.push_back(IntBoolPair(7, matchB1&&matchB2));

  // delta phi btw the b-jets

//   double deltaTheta = DeltaPhi<reco::Particle>()(solution.getJetB(), 
// 			solution.getJetBbar());
  
//   double deltaPhi = delta( solution.getJetB().p4().phi(),
// 		solution.getJetBbar().p4().phi() );
  double deltaTheta = abs( delta (solution.getJetBbar().p4().theta(),
		solution.getJetB().p4().theta() ) );

  evtselectVarVal.push_back(IntDblPair(8, deltaTheta));
  evtselectVarMatch.push_back(IntBoolPair(8, matchB1&&matchB2));

  //  Lower / Higher of phi difference between the b and associated lepton

  double deltaPhi1 = abs ( delta( solution.getJetB().p4().phi(),
		solution.getLeptPos().p4().phi() ) );
  double deltaPhi2 = abs ( delta( solution.getJetBbar().p4().phi(),
		solution.getLeptNeg().p4().phi() ) );
// if (deltaPhi1<0.05) {
// cout << deltaPhi1<<" "
// <<solution.getJetB().p4().phi()<<" "
// <<solution.getLeptPos().p4().phi()<<" "
// <<solution.getJetB().p4().eta()<<" "
// <<solution.getLeptPos().p4().eta()<<" "
// << solution.getJetB().p4().phi() - solution.getLeptPos().p4().phi()<<"\n";
// double d1 = solution.getJetB().p4().phi();
// double d2 = solution.getLeptPos().p4().phi();
// cout << deltaPhi1<<" "
// <<d1<<" "
// <<d2<<" "
// <<d1-d2<<" "
// <<delta(d1,d2)<<" "
// << solution.getJetB().p4().phi() - solution.getLeptPos().p4().phi()<<"\n";
// 
// }

  fillMinMax(deltaPhi1, deltaPhi2, 9, evtselectVarVal, 
	matchB&&matchLeptPos, matchBbar&&matchLeptNeg, evtselectVarMatch);

  //  Lower / Higher of theta difference between the b and associated lepton

  double deltaTheta1 = abs( solution.getJetB().p4().theta() -
		solution.getLeptPos().p4().theta() );
  double deltaTheta2 = abs( solution.getJetBbar().p4().theta() -
		solution.getLeptNeg().p4().theta() );
  fillMinMax(deltaTheta1, deltaTheta2, 11, evtselectVarVal, 
	matchB&&matchLeptPos, matchBbar&&matchLeptNeg, evtselectVarMatch);

  // Invariant Mass of lepton pair

  XYZTLorentzVector pp = solution.getLeptPos().p4() + solution.getLeptNeg().p4();
  double mass = pp.mass();
  evtselectVarVal.push_back(IntDblPair(13, mass));
  evtselectVarMatch.push_back(IntBoolPair(13, matchLeptNeg&&matchLeptPos));

  evtselectVarVal.push_back(IntDblPair(13, mass));
  evtselectVarMatch.push_back(IntBoolPair(13, matchLeptNeg&&matchLeptPos));

  vector <pat::Jet> jet3;
for (int i=0;i<jets->size();++i) {
if  ( ((*jets)[i].et()<solution.getJetB().et()) && ((*jets)[i].et()<solution.getJetBbar().et())) {jet3.push_back((*jets)[i]);
// cout << "jet " << i << " " << jet3.back().partonFlavour()<< " " << jet3.back().pt() << " " << jet3.back().eta()<<endl;
}}
  double jet1Ratio = 0., jet2Ratio = 0.;  
  if (jet3.size()>0) { 
    jet1Ratio = jet3[0].et()/solution.getJetB().et();
    jet2Ratio = jet3[0].et()/solution.getJetBbar().et();
  //cout << solution.getJetB().et() << " "<<solution.getJetBbar().et()<<endl;
  //cout << (*jets)[0].et() << " "<<(*jets)[1].et() << " "<<(*jets)[2].et() << jet1Ratio<< " "<<jet2Ratio<<endl<<endl;
  }
  fillMinMax(jet1Ratio, jet2Ratio, 14, evtselectVarVal, 
	matchB1, matchB2, evtselectVarMatch);

  evtselectVarVal.push_back(IntDblPair(16, jets->size()));
  evtselectVarMatch.push_back(IntBoolPair(16, matchB&&matchBbar));


  if (!matchOnly) solution.setLRSignalEvtObservables(evtselectVarVal);
  return evtselectVarMatch;
}

void TtDilepLRSignalSelObservables::fillMinMax
	(double v1, double v2, int obsNbr, vector< IntDblPair > & varList,
	 bool match1, bool match2, vector< IntBoolPair > & matchList)
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

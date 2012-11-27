#include "TopQuarkAnalysis/TopEventSelection/interface/TtSemiLRSignalSelObservables.h"
#include "FWCore/Utilities/interface/isFinite.h"

TtSemiLRSignalSelObservables::TtSemiLRSignalSelObservables()
{
}

TtSemiLRSignalSelObservables::~TtSemiLRSignalSelObservables()
{
}

void TtSemiLRSignalSelObservables::operator() (TtSemiEvtSolution &TS,const std::vector<pat::Jet>& SelectedTopJets)
{
  evtselectVarVal.clear();
  
  // choice the B-tag algorithm
  const char *BtagAlgo = "trackCountingJetTags";
  
  // activate the "debug mode"
  bool DEBUG = false;
  
  if(DEBUG) std::cout<<"---------- Start calculating the LR observables ----------"<<std::endl;
  
  
  std::vector<pat::Jet> TopJets;
  TopJets.clear();
  
  for(unsigned int i = 0 ; i<SelectedTopJets.size() ; i++)
    {
      TopJets.push_back(SelectedTopJets[i]);
    }
  
  //sort the TopJets in Et
  std::sort(TopJets.begin(),TopJets.end(),EtComparator);
  
  TLorentzVector * Hadp = new TLorentzVector();
  Hadp->SetPxPyPzE(TopJets[3].px(),TopJets[3].py(),TopJets[3].pz(),TopJets[3].energy());
  
  TLorentzVector * Hadq = new TLorentzVector();
  Hadq->SetPxPyPzE(TopJets[2].px(),TopJets[2].py(),TopJets[2].pz(),TopJets[2].energy());
  
  TLorentzVector * Hadb = new TLorentzVector();
  Hadb->SetPxPyPzE(TopJets[1].px(),TopJets[1].py(),TopJets[1].pz(),TopJets[1].energy());
  
  TLorentzVector * Lepb = new TLorentzVector();
  Lepb->SetPxPyPzE(TopJets[0].px(),TopJets[0].py(),TopJets[0].pz(),TopJets[0].energy());
  
  TLorentzVector * Lept = new TLorentzVector();
  if(TS.getDecay() == "muon") Lept->SetPxPyPzE(TS.getMuon().px(),TS.getMuon().py(),TS.getMuon().pz(),TS.getMuon().energy());
  else if(TS.getDecay() == "electron") Lept->SetPxPyPzE(TS.getElectron().px(),TS.getElectron().py(),TS.getElectron().pz(),TS.getElectron().energy());
  
  TLorentzVector *Lepn = new TLorentzVector();
  Lepn->SetPxPyPzE(TS.getNeutrino().px(),TS.getNeutrino().py(),TS.getNeutrino().pz(),TS.getNeutrino().energy());
  
  // Calculation of the pz of the neutrino due to W-mass constraint
  
  MEzCalculator *Mez = new MEzCalculator();
  Mez->SetMET(TS.getNeutrino());
  if(TS.getDecay() == "muon") Mez->SetLepton(TS.getMuon());
  else Mez->SetLepton(TS.getElectron(), false);
  double MEZ = Mez->Calculate();
  Lepn->SetPz(MEZ);
  
  // Pt of the lepton
  
  double Obs1 = Lept->Pt();
  evtselectVarVal.push_back(std::pair<unsigned int,double>(1,Obs1));
  if(DEBUG) std::cout<<"------ LR observable 1 "<<Obs1<<" calculated ------"<<std::endl;
  
  // Missing transverse energy
  
  double Obs2 = TS.getNeutrino().et();
  evtselectVarVal.push_back(std::pair<unsigned int,double>(2,Obs2));	
  if(DEBUG) std::cout<<"------ LR observable 2 "<<Obs2<<" calculated ------"<<std::endl;
  
  //HT variable (Et-sum of the four jets)
  
  double HT=0;
  for(unsigned int i=0;i<4;i++)
    {
      HT += TopJets[i].et();
    }
  
  double Obs3 = HT;
  evtselectVarVal.push_back(std::pair<unsigned int,double>(3,Obs3));
  if(DEBUG) std::cout<<"------ LR observable 3 "<<Obs3<<" calculated ------"<<std::endl;
  
  //Et-Sum of the lightest jets
  
  double EtSum = TopJets[2].et()+TopJets[3].et();
  double Obs4 = EtSum;
  evtselectVarVal.push_back(std::pair<unsigned int,double>(4,Obs4));
  if(DEBUG) std::cout<<"------ LR observable 4 "<<Obs4<<" calculated ------"<<std::endl;
  
  // Et-Ratio between the two lowest jets in Et and four highest jets
  
  double Obs5 = EtSum/HT;
  evtselectVarVal.push_back(std::pair<unsigned int,double>(5,Obs5));
  if(DEBUG) std::cout<<"------ LR observable 5 "<<Obs5<<" calculated ------"<<std::endl;
  
  // Et-Ratio between the two highest jets in Et and four highest jets
  
  double Obs6 = (HT-EtSum)/HT;
  evtselectVarVal.push_back(std::pair<unsigned int,double>(6,Obs6));
  if(DEBUG) std::cout<<"------ LR observable 6 "<<Obs6<<" calculated ------"<<std::endl;
  
  // Transverse Mass of the system
  
  TLorentzVector TtbarSystem = (*Hadp)+(*Hadq)+(*Hadb)+(*Lepb)+(*Lept)+(*Lepn);
  double MT = TtbarSystem.Mt();
  double Obs7 = MT;
  evtselectVarVal.push_back(std::pair<unsigned int,double>(7,Obs7));
  if(DEBUG) std::cout<<"------ LR observable 7 "<<Obs7<<" calculated ------"<<std::endl;
  
  // Observables related to the b-disc (jets ordered in Bdisc)
  
  //sort the TopJets in Bdiscriminant
  std::sort(TopJets.begin(),TopJets.end(),BdiscComparator);
  
  double Obs8;
  double Obs9;	
  double Obs10;
  double Obs11;	
  
  // Difference in bdisc between the 2nd and the 3rd jets
  double BGap = TopJets[1].bDiscriminator(BtagAlgo) - TopJets[2].bDiscriminator(BtagAlgo);
  
  // Sum of the bdisc of the two highest/lowest jets
  double BjetsBdiscSum = TopJets[0].bDiscriminator(BtagAlgo) + TopJets[1].bDiscriminator(BtagAlgo);
  double LjetsBdiscSum = TopJets[2].bDiscriminator(BtagAlgo) + TopJets[3].bDiscriminator(BtagAlgo);
  
  Obs8  = (TopJets[2].bDiscriminator(BtagAlgo) > -10 ? log(BGap) : -5);
  if(DEBUG) std::cout<<"------ LR observable 8 "<<Obs8<<" calculated ------"<<std::endl;
  Obs9  = (BjetsBdiscSum*BGap);	
  if(DEBUG) std::cout<<"------ LR observable 9 "<<Obs9<<" calculated ------"<<std::endl;
  Obs10 = (BjetsBdiscSum/LjetsBdiscSum);
  if(DEBUG) std::cout<<"------ LR observable 10 "<<Obs10<<" calculated ------"<<std::endl;
  // Distance from the origin in the (BjetsBdiscSum, LjetsBdiscSum) plane
  Obs11 = 0.707*((BjetsBdiscSum+LjetsBdiscSum)/2 +10);	
  if(DEBUG) std::cout<<"------ LR observable 11 "<<Obs11<<" calculated ------"<<std::endl;
  
  
  evtselectVarVal.push_back(std::pair<unsigned int,double>(8,Obs8));
  evtselectVarVal.push_back(std::pair<unsigned int,double>(9,Obs9));
  evtselectVarVal.push_back(std::pair<unsigned int,double>(10,Obs10));
  evtselectVarVal.push_back(std::pair<unsigned int,double>(11,Obs11));
  
  //sort the TopJets in Et
  std::sort(TopJets.begin(),TopJets.end(),EtComparator);
  
  // Circularity of the event
  
  double N=0,D=0,C_tmp=0,C=10000;
  double N_NoNu=0,D_NoNu=0,C_NoNu_tmp=0,C_NoNu=10000;
  double nx,ny,nz;
  
  // C = 2min(E(pt.n)^2/E(pt)^2) = 2*N/D but it is theorically preferable to use C'=PI/2*min(E|pt.n|/E|pt|), sum over all jets+lepton+MET (cf PhysRevD 48 R3953(Nov 1993))
  
  for(unsigned int i=0;i<4;i++)
    {
      D += fabs(TopJets[i].pt());
    }
  // modified the 26th of September to calculate also the circularity without the neutrino contribution
  D += fabs(Lept->Pt());
  D_NoNu = D;
  D += fabs(Lepn->Pt());
  
  if((D>0))
    {
      
      // Loop over all the unit vectors in the transverse plane in order to find the miminum : 
      for(unsigned int i=0; i<720; i++)
	{
	  
	  nx = cos((2*PI/720)*i);
	  ny = sin((2*PI/720)*i);
	  nz = 0;
	  N=0;
	  
	  for(unsigned int i=0;i<4;i++)
	    {
	      N += fabs(TopJets[i].px()*nx+TopJets[i].py()*ny+TopJets[i].pz()*nz);
	    }
	  // modified the 26th of September to calculate also the circularity without the neutrino contribution
	  N += fabs(Lept->Px()*nx+Lept->Py()*ny+Lept->Pz()*nz);
	  N_NoNu = N;
	  N += fabs(Lepn->Px()*nx+Lepn->Py()*ny+Lepn->Pz()*nz);
	  
	  C_tmp = 2*N/D;
	  
	  // modified the 26th of September to calculate also the circularity without the neutrino contribution
	  C_NoNu_tmp = 2*N_NoNu/D_NoNu;
	  
	  if(C_tmp<C) C = C_tmp;
	  
	  // modified the 26th of September to calculate also the circularity without the neutrino contribution
	  if(C_NoNu_tmp<C_NoNu) C_NoNu = C_NoNu_tmp;
	}
    }
  
  double Obs12 = ( C!=10000 ? C : 0);
  evtselectVarVal.push_back(std::pair<unsigned int,double>(12,Obs12));
  if(DEBUG) std::cout<<"------ LR observable 12 "<<Obs12<<" calculated ------"<<std::endl;
  
  // Circularity of the event without neutrino
  
  double Obs13 = ( C_NoNu != 10000 ? C_NoNu : 0);
  evtselectVarVal.push_back(std::pair<unsigned int,double>(13,Obs13));
  if(DEBUG) std::cout<<"------ LR observable 13 "<<Obs13<<" calculated ------"<<std::endl;
  
  // Centrality of the four highest jets
  
  double H=0;
  for(unsigned int i=0;i<4;i++)
    {
      H += TopJets[i].energy();
    }
  
  double Obs14 = HT/H;
  evtselectVarVal.push_back(std::pair<unsigned int,double>(14,Obs14));
  if(DEBUG) std::cout<<"------ LR observable 14 "<<Obs14<<" calculated ------"<<std::endl;
  
  // Sphericity and Aplanarity without boosting back the system to CM frame
  
  TMatrixDSym Matrix(3);
  
  double PX2 = std::pow(Hadp->Px(),2)+std::pow(Hadq->Px(),2)+std::pow(Hadb->Px(),2)+std::pow(Lepb->Px(),2)+std::pow(Lept->Px(),2)+std::pow(Lepn->Px(),2);
  double PY2 = std::pow(Hadp->Py(),2)+std::pow(Hadq->Py(),2)+std::pow(Hadb->Py(),2)+std::pow(Lepb->Py(),2)+std::pow(Lept->Py(),2)+std::pow(Lepn->Py(),2);
  double PZ2 = std::pow(Hadp->Pz(),2)+std::pow(Hadq->Pz(),2)+std::pow(Hadb->Pz(),2)+std::pow(Lepb->Pz(),2)+std::pow(Lept->Pz(),2)+std::pow(Lepn->Pz(),2);
  
  double P2  = PX2+PY2+PZ2;
  
  double PXY = Hadp->Px()*Hadp->Py()+Hadq->Px()*Hadq->Py()+Hadb->Px()*Hadb->Py()+Lepb->Px()*Lepb->Py()+Lept->Px()*Lept->Py()+Lepn->Px()*Lepn->Py();
  double PXZ = Hadp->Px()*Hadp->Pz()+Hadq->Px()*Hadq->Pz()+Hadb->Px()*Hadb->Pz()+Lepb->Px()*Lepb->Pz()+Lept->Px()*Lept->Pz()+Lepn->Px()*Lepn->Pz();
  double PYZ = Hadp->Py()*Hadp->Pz()+Hadq->Py()*Hadq->Pz()+Hadb->Py()*Hadb->Pz()+Lepb->Py()*Lepb->Pz()+Lept->Py()*Lept->Pz()+Lepn->Py()*Lepn->Pz();
  
  Matrix(0,0) = PX2/P2; Matrix(0,1) = PXY/P2; Matrix(0,2) = PXZ/P2;
  Matrix(1,0) = PXY/P2; Matrix(1,1) = PY2/P2; Matrix(1,2) = PYZ/P2;
  Matrix(2,0) = PXZ/P2; Matrix(2,1) = PYZ/P2; Matrix(2,2) = PZ2/P2;
  
  TMatrixDSymEigen pTensor(Matrix);
  
  std::vector<double> EigValues;
  EigValues.clear();
  for(int i=0;i<3;i++)
    {
      EigValues.push_back(pTensor.GetEigenValues()[i]);
    }
  
  std::sort(EigValues.begin(),EigValues.end(),dComparator);
  
  double Sphericity = 1.5*(EigValues[1]+EigValues[2]);
  double Aplanarity = 1.5*EigValues[2];
  
  double Obs15 = (edm::isNotFinite(Sphericity) ? 0 : Sphericity);
  evtselectVarVal.push_back(std::pair<unsigned int,double>(15,Obs15));
  if(DEBUG) std::cout<<"------ LR observable 15 "<<Obs15<<" calculated ------"<<std::endl;
  
  double Obs16 = (edm::isNotFinite(Aplanarity) ? 0 : Aplanarity);
  evtselectVarVal.push_back(std::pair<unsigned int,double>(16,Obs16));
  if(DEBUG) std::cout<<"------ LR observable 16 "<<Obs16<<" calculated ------"<<std::endl;
  
  
  // Sphericity and Aplanarity with boosting back the system to CM frame	
  
  TVector3 BoostBackToCM = -(TtbarSystem.BoostVector());
  Hadp->Boost(BoostBackToCM);
  Hadq->Boost(BoostBackToCM);
  Hadb->Boost(BoostBackToCM);
  Lepb->Boost(BoostBackToCM);
  Lept->Boost(BoostBackToCM);
  Lepn->Boost(BoostBackToCM);
  
  
  double BOOST_PX2 = std::pow(Hadp->Px(),2)+std::pow(Hadq->Px(),2)+std::pow(Hadb->Px(),2)+std::pow(Lepb->Px(),2)+std::pow(Lept->Px(),2)+std::pow(Lepn->Px(),2);
  double BOOST_PY2 = std::pow(Hadp->Py(),2)+std::pow(Hadq->Py(),2)+std::pow(Hadb->Py(),2)+std::pow(Lepb->Py(),2)+std::pow(Lept->Py(),2)+std::pow(Lepn->Py(),2);
  double BOOST_PZ2 = std::pow(Hadp->Pz(),2)+std::pow(Hadq->Pz(),2)+std::pow(Hadb->Pz(),2)+std::pow(Lepb->Pz(),2)+std::pow(Lept->Pz(),2)+std::pow(Lepn->Pz(),2);
  
  double BOOST_P2  = BOOST_PX2+BOOST_PY2+BOOST_PZ2;
  
  double BOOST_PXY = Hadp->Px()*Hadp->Py()+Hadq->Px()*Hadq->Py()+Hadb->Px()*Hadb->Py()+Lepb->Px()*Lepb->Py()+Lept->Px()*Lept->Py()+Lepn->Px()*Lepn->Py();
  double BOOST_PXZ = Hadp->Px()*Hadp->Pz()+Hadq->Px()*Hadq->Pz()+Hadb->Px()*Hadb->Pz()+Lepb->Px()*Lepb->Pz()+Lept->Px()*Lept->Pz()+Lepn->Px()*Lepn->Pz();
  double BOOST_PYZ = Hadp->Py()*Hadp->Pz()+Hadq->Py()*Hadq->Pz()+Hadb->Py()*Hadb->Pz()+Lepb->Py()*Lepb->Pz()+Lept->Py()*Lept->Pz()+Lepn->Py()*Lepn->Pz();
  
  TMatrixDSym BOOST_Matrix(3);
  
  BOOST_Matrix(0,0) = BOOST_PX2/BOOST_P2; BOOST_Matrix(0,1) = BOOST_PXY/BOOST_P2; BOOST_Matrix(0,2) = BOOST_PXZ/BOOST_P2;
  BOOST_Matrix(1,0) = BOOST_PXY/BOOST_P2; BOOST_Matrix(1,1) = BOOST_PY2/BOOST_P2; BOOST_Matrix(1,2) = BOOST_PYZ/BOOST_P2;
  BOOST_Matrix(2,0) = BOOST_PXZ/BOOST_P2; BOOST_Matrix(2,1) = BOOST_PYZ/BOOST_P2; BOOST_Matrix(2,2) = BOOST_PZ2/BOOST_P2;
  
  TMatrixDSymEigen BOOST_pTensor(BOOST_Matrix);
  
  std::vector<double> BOOST_EigValues;
  BOOST_EigValues.clear();
  for(int i=0;i<3;i++)
    {
      BOOST_EigValues.push_back(BOOST_pTensor.GetEigenValues()[i]);
    }
  
  std::sort(BOOST_EigValues.begin(),BOOST_EigValues.end(),dComparator);
  
  double BOOST_Sphericity = 1.5*(BOOST_EigValues[1]+BOOST_EigValues[2]);
  double BOOST_Aplanarity = 1.5*BOOST_EigValues[2];
  
  double Obs17 = ( edm::isNotFinite(BOOST_Sphericity) ? 0 : BOOST_Sphericity );
  evtselectVarVal.push_back(std::pair<unsigned int,double>(17,Obs17));
  if(DEBUG) std::cout<<"------ LR observable 17 "<<Obs17<<" calculated ------"<<std::endl;
  
  double Obs18 = ( edm::isNotFinite(BOOST_Aplanarity) ? 0 : BOOST_Aplanarity );
  evtselectVarVal.push_back(std::pair<unsigned int,double>(18,Obs18));
  if(DEBUG) std::cout<<"------ LR observable 18 "<<Obs18<<" calculated ------"<<std::endl;
  
  //ratio between ET of the fifth jet and HT
  
  double Obs19 = (TopJets.size() > 4) ? TopJets[4].et()/HT : 1.0;
  evtselectVarVal.push_back(std::pair<unsigned int,double>(19,Obs19));
  if(DEBUG) std::cout<<"------ LR observable 19 "<<Obs19<<" calculated ------"<<std::endl;
  
  // HT variable calculated with all the jets in the event.
  
  double HT_alljets = 0;
  double  H_alljets = 0;
  for(unsigned int i=0;i<TopJets.size();i++)
    {
      HT_alljets += TopJets[i].et();
      H_alljets  += TopJets[i].energy();
    }
  double Obs20 = HT_alljets;
  evtselectVarVal.push_back(std::pair<unsigned int,double>(20,Obs20));
  if(DEBUG) std::cout<<"------ LR observable 20 "<<Obs20<<" calculated ------"<<std::endl;
  
  // HT3 = HT calculated with all jets except the two leading jets
  
  double HT3_alljets = 0;
  for(unsigned int i=2;i<TopJets.size();i++)
    {
      HT3_alljets += TopJets[i].et();
    }
  double Obs21 = HT3_alljets;
  evtselectVarVal.push_back(std::pair<unsigned int,double>(21,Obs21));
  if(DEBUG) std::cout<<"------ LR observable 21 "<<Obs21<<" calculated ------"<<std::endl;
  
  // ET0, ratio of the Et of the leading and HT_alljets
  
  double ET0 = TopJets[0].et()/HT_alljets;
  double Obs22 = ET0;
  evtselectVarVal.push_back(std::pair<unsigned int,double>(22,Obs22));
  if(DEBUG) std::cout<<"------ LR observable 22 "<<Obs22<<" calculated ------"<<std::endl;
  
  // Centrality of the event computed with all jets
  
  double Obs23 = ( (H_alljets>0) ? HT_alljets/H_alljets : 0);
  evtselectVarVal.push_back(std::pair<unsigned int,double>(23,Obs23));
  if(DEBUG) std::cout<<"------ LR observable 23 "<<Obs23<<" calculated ------"<<std::endl;
  
  //Fox-Wolfram momenta (1st to 6th), modified for hadron collider and using a Legendre polynomials expansion
  
  double FW_momentum_0=0, FW_momentum_1=0, FW_momentum_2=0, FW_momentum_3=0, FW_momentum_4=0, FW_momentum_5=0, FW_momentum_6=0;
  
  for(unsigned int i=0;i<TopJets.size();i++)
    {
      for(unsigned int j=0;j<TopJets.size();j++)
	{
	  double ET_ij_over_ETSum2= TopJets[i].et()*TopJets[j].et()/(std::pow(HT_alljets,2));
	  double cosTheta_ij = (TopJets[i].px()*TopJets[j].px()+
				TopJets[i].py()*TopJets[j].py()+
				TopJets[i].pz()*TopJets[j].pz())
	    /(TopJets[i].p4().R()*TopJets[j].p4().R());
	  FW_momentum_0 += ET_ij_over_ETSum2;
	  FW_momentum_1 += ET_ij_over_ETSum2 * cosTheta_ij;
	  FW_momentum_2 += ET_ij_over_ETSum2 * 0.5   * (  3*std::pow(cosTheta_ij,2)- 1);
	  FW_momentum_3 += ET_ij_over_ETSum2 * 0.5   * (  5*std::pow(cosTheta_ij,3)-  3*cosTheta_ij);
	  FW_momentum_4 += ET_ij_over_ETSum2 * 0.125 * ( 35*std::pow(cosTheta_ij,4)- 30*std::pow(cosTheta_ij,2)+3);
	  FW_momentum_5 += ET_ij_over_ETSum2 * 0.125 * ( 63*std::pow(cosTheta_ij,5)- 70*std::pow(cosTheta_ij,3)+15*cosTheta_ij);
	  FW_momentum_6 += ET_ij_over_ETSum2 * 0.0625* (231*std::pow(cosTheta_ij,6)-315*std::pow(cosTheta_ij,4)+105*std::pow(cosTheta_ij,2)-5);
	}
    }
  
  double Obs24 = FW_momentum_0;
  evtselectVarVal.push_back(std::pair<unsigned int,double>(24,Obs24));
  if(DEBUG) std::cout<<"------ LR observable 24 "<<Obs24<<" calculated ------"<<std::endl;
  double Obs25 = FW_momentum_1;
  evtselectVarVal.push_back(std::pair<unsigned int,double>(25,Obs25));
  if(DEBUG) std::cout<<"------ LR observable 25 "<<Obs25<<" calculated ------"<<std::endl;
  double Obs26 = FW_momentum_2;
  evtselectVarVal.push_back(std::pair<unsigned int,double>(26,Obs26));
  if(DEBUG) std::cout<<"------ LR observable 26 "<<Obs26<<" calculated ------"<<std::endl;
  double Obs27 = FW_momentum_3;
  evtselectVarVal.push_back(std::pair<unsigned int,double>(27,Obs27));
  if(DEBUG) std::cout<<"------ LR observable 27 "<<Obs27<<" calculated ------"<<std::endl;
  double Obs28 = FW_momentum_4;
  evtselectVarVal.push_back(std::pair<unsigned int,double>(28,Obs28));
  if(DEBUG) std::cout<<"------ LR observable 28 "<<Obs28<<" calculated ------"<<std::endl;
  double Obs29 = FW_momentum_5;
  evtselectVarVal.push_back(std::pair<unsigned int,double>(29,Obs29));
  if(DEBUG) std::cout<<"------ LR observable 29 "<<Obs29<<" calculated ------"<<std::endl;
  double Obs30 = FW_momentum_6;
  evtselectVarVal.push_back(std::pair<unsigned int,double>(30,Obs30));
  if(DEBUG) std::cout<<"------ LR observable 30 "<<Obs30<<" calculated ------"<<std::endl;
  
  // Thrust, thrust major and thrust minor
  
  TVector3 n(0,0,0), n_tmp(0,0,0);
  
  double Thrust_D = 0,      Thrust_N = 0;
  double Thrust = -1,       Thrust_tmp =0;
  double Thrust_Major = -1, Thrust_Major_tmp =0;
  double Thrust_Minor = -1;
  
  for(unsigned int i=0;i<TopJets.size();i++)
    {
      Thrust_D += TopJets[i].p();
    }
  
  Thrust_D += Lept->P();
  
  if(Thrust_D>0)
    {
      // Calculation of the thrust :
      for(unsigned int i=0;i<720;i++)
	{
	  for(unsigned int j=0;j<360;j++)
	    {
	      n_tmp.SetX(cos((2*PI/720)*i)*sin((PI/360)*j));
	      n_tmp.SetY(sin((2*PI/720)*i)*sin((PI/360)*j));
	      n_tmp.SetZ(cos((PI/360)*j));
	      
	      for(unsigned int k=0;k<TopJets.size();k++)
		{
		  Thrust_N += fabs(TopJets[k].px()*(n_tmp.x())+TopJets[k].py()*(n_tmp.y())+TopJets[k].pz()*(n_tmp.z()));
		}
	      Thrust_N += fabs(Lept->Px()*(n_tmp.x())+Lept->Py()*(n_tmp.y())+Lept->Pz()*(n_tmp.z()));
	      
	      Thrust_tmp = Thrust_N/Thrust_D;
	      
	      Thrust_N = 0;
	      if(Thrust_tmp > Thrust)
		{
		  Thrust = Thrust_tmp;
		  n.SetXYZ(n_tmp.x(),n_tmp.y(),n_tmp.z());
		}	
	    }
	}
      
      // Calculation of the thrust major :
      TVector3 nT = n.Orthogonal();
      nT = nT.Unit();
      for(unsigned int i=0;i<720;i++)
	{
	  nT.Rotate((2*PI/720)*i,n);
	  for(unsigned int j=0;j<TopJets.size();j++)
	    {
	      Thrust_N += fabs(TopJets[j].px()*(nT.x())+TopJets[j].py()*(nT.y())+TopJets[j].pz()*(nT.z()));
	    }
	  Thrust_N += fabs(Lept->Px()*nT.x()+Lept->Py()*(nT.y())+Lept->Pz()*(nT.z()));
	  
	  Thrust_Major_tmp = Thrust_N/Thrust_D;
	  Thrust_N = 0;
	  
	  if(Thrust_Major_tmp > Thrust_Major)
	    {
	      Thrust_Major = Thrust_Major_tmp;
	    }
	}
      
      // Calculation of the thrust minor :
      
      TVector3 nMinor = nT.Cross(n);
      nMinor = nMinor.Unit();
      
      for(unsigned int i=0;i<TopJets.size();i++)
	{
	  Thrust_N += fabs(TopJets[i].px()*(nMinor.x())+TopJets[i].py()*(nMinor.y())+TopJets[i].pz()*(nMinor.z()));
	}
      Thrust_N += fabs(Lept->Px()*nMinor.x()+Lept->Py()*(nMinor.y())+Lept->Pz()*(nMinor.z()));
      
      Thrust_Minor =  Thrust_N/Thrust_D;
    }
  
  double Obs31 = Thrust;
  evtselectVarVal.push_back(std::pair<unsigned int,double>(31,Obs31));
  if(DEBUG) std::cout<<"------ LR observable 31 "<<Obs31<<" calculated ------"<<std::endl;
  double Obs32 = Thrust_Major;
  evtselectVarVal.push_back(std::pair<unsigned int,double>(32,Obs32));
  if(DEBUG) std::cout<<"------ LR observable 32 "<<Obs32<<" calculated ------"<<std::endl;
  double Obs33 = Thrust_Minor;
  evtselectVarVal.push_back(std::pair<unsigned int,double>(33,Obs33));
  if(DEBUG) std::cout<<"------ LR observable 33 "<<Obs33<<" calculated ------"<<std::endl;
  
  // Oblateness
  
  double Obs34 = Thrust_Major-Thrust_Minor;
  evtselectVarVal.push_back(std::pair<unsigned int,double>(34,Obs34));
  if(DEBUG) std::cout<<"------ LR observable 34 "<<Obs34<<" calculated ------"<<std::endl;
  
  // Sphericity and Aplanarity without boosting back the system to CM frame and without neutrino
  
  TMatrixDSym Matrix_NoNu(3);
  
  double PX2_NoNu = std::pow(Hadp->Px(),2)+std::pow(Hadq->Px(),2)+std::pow(Hadb->Px(),2)+std::pow(Lepb->Px(),2)+std::pow(Lept->Px(),2);
  double PY2_NoNu = std::pow(Hadp->Py(),2)+std::pow(Hadq->Py(),2)+std::pow(Hadb->Py(),2)+std::pow(Lepb->Py(),2)+std::pow(Lept->Py(),2);
  double PZ2_NoNu = std::pow(Hadp->Pz(),2)+std::pow(Hadq->Pz(),2)+std::pow(Hadb->Pz(),2)+std::pow(Lepb->Pz(),2)+std::pow(Lept->Pz(),2);
  
  double P2_NoNu  = PX2_NoNu+PY2_NoNu+PZ2_NoNu;
  
  double PXY_NoNu = Hadp->Px()*Hadp->Py()+Hadq->Px()*Hadq->Py()+Hadb->Px()*Hadb->Py()+Lepb->Px()*Lepb->Py()+Lept->Px()*Lept->Py();
  double PXZ_NoNu = Hadp->Px()*Hadp->Pz()+Hadq->Px()*Hadq->Pz()+Hadb->Px()*Hadb->Pz()+Lepb->Px()*Lepb->Pz()+Lept->Px()*Lept->Pz();
  double PYZ_NoNu = Hadp->Py()*Hadp->Pz()+Hadq->Py()*Hadq->Pz()+Hadb->Py()*Hadb->Pz()+Lepb->Py()*Lepb->Pz()+Lept->Py()*Lept->Pz();
  
  Matrix_NoNu(0,0) = PX2_NoNu/P2_NoNu; Matrix_NoNu(0,1) = PXY_NoNu/P2_NoNu; Matrix_NoNu(0,2) = PXZ_NoNu/P2_NoNu;
  Matrix_NoNu(1,0) = PXY_NoNu/P2_NoNu; Matrix_NoNu(1,1) = PY2_NoNu/P2_NoNu; Matrix_NoNu(1,2) = PYZ_NoNu/P2_NoNu;
  Matrix_NoNu(2,0) = PXZ_NoNu/P2_NoNu; Matrix_NoNu(2,1) = PYZ_NoNu/P2_NoNu; Matrix_NoNu(2,2) = PZ2_NoNu/P2_NoNu;
  
  TMatrixDSymEigen pTensor_NoNu(Matrix_NoNu);
  
  std::vector<double> EigValues_NoNu;
  EigValues_NoNu.clear();
  for(int i=0;i<3;i++)
    {
      EigValues_NoNu.push_back(pTensor_NoNu.GetEigenValues()[i]);
    }
  
  std::sort(EigValues_NoNu.begin(),EigValues_NoNu.end(),dComparator);
  
  double Sphericity_NoNu = 1.5*(EigValues_NoNu[1]+EigValues_NoNu[2]);
  double Aplanarity_NoNu = 1.5*EigValues_NoNu[2];
  
  double Obs35 = (edm::isNotFinite(Sphericity_NoNu) ? 0 : Sphericity_NoNu);
  evtselectVarVal.push_back(std::pair<unsigned int,double>(35,Obs35));
  if(DEBUG) std::cout<<"------ LR observable 35 "<<Obs35<<" calculated ------"<<std::endl;
  
  double Obs36 = (edm::isNotFinite(Aplanarity_NoNu) ? 0 : Aplanarity_NoNu);
  evtselectVarVal.push_back(std::pair<unsigned int,double>(36,Obs36));
  if(DEBUG) std::cout<<"------ LR observable 36 "<<Obs36<<" calculated ------"<<std::endl;
  
  // Sphericity and Aplanarity without boosting back the system to CM frame and without neutrino or lepton
  
  TMatrixDSym Matrix_NoNuNoLep(3);
  
  double PX2_NoNuNoLep = std::pow(Hadp->Px(),2)+std::pow(Hadq->Px(),2)+std::pow(Hadb->Px(),2)+std::pow(Lepb->Px(),2);
  double PY2_NoNuNoLep = std::pow(Hadp->Py(),2)+std::pow(Hadq->Py(),2)+std::pow(Hadb->Py(),2)+std::pow(Lepb->Py(),2);
  double PZ2_NoNuNoLep = std::pow(Hadp->Pz(),2)+std::pow(Hadq->Pz(),2)+std::pow(Hadb->Pz(),2)+std::pow(Lepb->Pz(),2);
  
  double P2_NoNuNoLep  = PX2_NoNuNoLep+PY2_NoNuNoLep+PZ2_NoNuNoLep;
  
  double PXY_NoNuNoLep = Hadp->Px()*Hadp->Py()+Hadq->Px()*Hadq->Py()+Hadb->Px()*Hadb->Py()+Lepb->Px()*Lepb->Py();
  double PXZ_NoNuNoLep = Hadp->Px()*Hadp->Pz()+Hadq->Px()*Hadq->Pz()+Hadb->Px()*Hadb->Pz()+Lepb->Px()*Lepb->Pz();
  double PYZ_NoNuNoLep = Hadp->Py()*Hadp->Pz()+Hadq->Py()*Hadq->Pz()+Hadb->Py()*Hadb->Pz()+Lepb->Py()*Lepb->Pz();
  
  Matrix_NoNuNoLep(0,0) = PX2_NoNuNoLep/P2_NoNuNoLep; Matrix_NoNuNoLep(0,1) = PXY_NoNuNoLep/P2_NoNuNoLep; Matrix_NoNuNoLep(0,2) = PXZ_NoNuNoLep/P2_NoNuNoLep;
  Matrix_NoNuNoLep(1,0) = PXY_NoNuNoLep/P2_NoNuNoLep; Matrix_NoNuNoLep(1,1) = PY2_NoNuNoLep/P2_NoNuNoLep; Matrix_NoNuNoLep(1,2) = PYZ_NoNuNoLep/P2_NoNuNoLep;
  Matrix_NoNuNoLep(2,0) = PXZ_NoNuNoLep/P2_NoNuNoLep; Matrix_NoNuNoLep(2,1) = PYZ_NoNuNoLep/P2_NoNuNoLep; Matrix_NoNuNoLep(2,2) = PZ2_NoNuNoLep/P2_NoNuNoLep;
  
  TMatrixDSymEigen pTensor_NoNuNoLep(Matrix_NoNuNoLep);
  
  std::vector<double> EigValues_NoNuNoLep;
  EigValues_NoNuNoLep.clear();
  for(int i=0;i<3;i++)
    {
      EigValues_NoNuNoLep.push_back(pTensor_NoNuNoLep.GetEigenValues()[i]);
    }
  
  std::sort(EigValues_NoNuNoLep.begin(),EigValues_NoNuNoLep.end(),dComparator);
  
  double Sphericity_NoNuNoLep = 1.5*(EigValues_NoNuNoLep[1]+EigValues_NoNuNoLep[2]);
  double Aplanarity_NoNuNoLep = 1.5*EigValues_NoNuNoLep[2];
  
  double Obs37 = (edm::isNotFinite(Sphericity_NoNuNoLep) ? 0 : Sphericity_NoNuNoLep);
  evtselectVarVal.push_back(std::pair<unsigned int,double>(37,Obs37));
  if(DEBUG) std::cout<<"------ LR observable 37 "<<Obs37<<" calculated ------"<<std::endl;
  
  double Obs38 = (edm::isNotFinite(Aplanarity_NoNuNoLep) ? 0 : Aplanarity_NoNuNoLep);
  evtselectVarVal.push_back(std::pair<unsigned int,double>(38,Obs38));
  if(DEBUG) std::cout<<"------ LR observable 38 "<<Obs38<<" calculated ------"<<std::endl;
  
  
  // Put the vector in the TtSemiEvtSolution
  TS.setLRSignalEvtObservables(evtselectVarVal);
  if(DEBUG) std::cout<<"------  Observable values stored in the event  ------"<<std::endl;
  
  delete Hadp;
  if(DEBUG) std::cout<<"------     Pointer to Hadp deleted             ------"<<std::endl;
  delete Hadq;
  if(DEBUG) std::cout<<"------     Pointer to Hadq deleted             ------"<<std::endl;
  delete Hadb;
  if(DEBUG) std::cout<<"------     Pointer to Hadb deleted             ------"<<std::endl;
  delete Lepb;
  if(DEBUG) std::cout<<"------     Pointer to Lepb deleted      ------"<<std::endl;
  delete Lept;
  if(DEBUG) std::cout<<"------     Pointer to Lepn deleted      ------"<<std::endl;
  delete Lepn;
  if(DEBUG) std::cout<<"------     Pointer to Mez deleted       ------"<<std::endl;
  delete Mez;
  
  if(DEBUG) std::cout<<"------------ LR observables calculated -----------"<<std::endl;  
}

#include "TopQuarkAnalysis/TopEventSelection/interface/TtSemiLRSignalSelObservables.h"

using namespace reco;
using namespace std;
using namespace math;


/************** Definition of the functions of the class ***************/

//Constructor
TtSemiLRSignalSelObservables::TtSemiLRSignalSelObservables(){}
// Destructor
TtSemiLRSignalSelObservables::~TtSemiLRSignalSelObservables(){}

void TtSemiLRSignalSelObservables::operator() (TtSemiEvtSolution &TS)
{
	evtselectVarVal.clear();
	
	//cout<<"New event being processed"<<endl;
	
	vector<TopJet> TopJets;
	TopJets.clear();
	TopJets.push_back(TS.getHadp());
	TopJets.push_back(TS.getHadq());
	TopJets.push_back(TS.getHadb());
	TopJets.push_back(TS.getLepb());

	//sort the TopJets in Et
	std::sort(TopJets.begin(),TopJets.end(),EtComparator);

// Calculation of the pz of the neutrino due to W-mass constraint

	TLorentzVector *Hadp = new TLorentzVector();
	Hadp->SetPxPyPzE(TopJets[3].getRecJet().px(),TopJets[3].getRecJet().py(),TopJets[3].getRecJet().pz(),TopJets[3].getRecJet().energy());
	
	TLorentzVector *Hadq = new TLorentzVector();
	Hadp->SetPxPyPzE(TopJets[2].getRecJet().px(),TopJets[2].getRecJet().py(),TopJets[2].getRecJet().pz(),TopJets[2].getRecJet().energy());

	TLorentzVector *Hadb = new TLorentzVector();
	Hadp->SetPxPyPzE(TopJets[1].getRecJet().px(),TopJets[1].getRecJet().py(),TopJets[1].getRecJet().pz(),TopJets[1].getRecJet().energy());

	TLorentzVector *Lepb = new TLorentzVector();
	Lepb->SetPxPyPzE(TopJets[0].getRecJet().px(),TopJets[0].getRecJet().py(),TopJets[0].getRecJet().pz(),TopJets[0].getRecJet().energy());
	
	TLorentzVector *Lept = new TLorentzVector();
	Lept->SetPxPyPzE(TS.getRecLept().px(),TS.getRecLept().py(),TS.getRecLept().pz(),TS.getRecLept().energy());
	
	TLorentzVector *Lepn = new TLorentzVector();
	Lepn->SetPxPyPzE(TS.getRecLepn().px(),TS.getRecLepn().py(),TS.getRecLepn().pz(),TS.getRecLepn().energy());
	
	double alpha = pow(Lept->E(),2)-pow(Lept->Pz(),2);
	double zeta  = pow(Lept->E(),2)-pow(80.41,2)-2*(Lept->Px()*Lepn->Px()+Lept->Py()*Lepn->Py());
	double beta  = Lept->Pz()*zeta;	
	double gamma = pow(Lept->E(),2)*(pow(Lepn->Pt(),2))-pow(zeta/2,2);	
	double Delta = pow(beta,2)-4*alpha*gamma;
	double LepnPt= Lepn->Pt();
	int    it    = 0;	

	while(Delta<0 && it<1000)
	{
		LepnPt = LepnPt-0.1;
		gamma = pow(Lept->E(),2)*(pow(LepnPt,2))-pow(zeta/2,2);
		Delta = pow(beta,2)-4*alpha*gamma;
		//cout<<"Look for another solution, Pt of the neutrino over estimated"<<endl;
		it++;
	}

	double Solution1 = (-beta-sqrt(Delta))/(2*alpha);
	double Solution2 = (-beta+sqrt(Delta))/(2*alpha);
	
	Lepn->SetPz(Solution1);	
	TLorentzVector *LepTop1 = new TLorentzVector();
	LepTop1->SetPxPyPzE(Lepb->Px()+Lept->Px()+Lepn->Px(),Lepb->Py()+Lept->Py()+Lepn->Py(),Lepb->Pz()+Lept->Pz()+Solution1,Lepb->E()+Lept->E()+Lepn->E());	
	double DiffLepTopMass1 = fabs(LepTop1->M()-175);
	
	Lepn->SetPz(Solution2);	
	TLorentzVector *LepTop2 = new TLorentzVector();
	LepTop2->SetPxPyPzE(Lepb->Px()+Lept->Px()+Lepn->Px(),Lepb->Py()+Lept->Py()+Lepn->Py(),Lepb->Pz()+Lept->Pz()+Solution2,Lepb->E()+Lept->E()+Lepn->E());		
	double DiffLepTopMass2 = fabs(LepTop2->M()-175);
	
	double Solution = (DiffLepTopMass1<DiffLepTopMass2 ? Solution1 : Solution2);
	
	Lepn->SetPz(Solution);
	
	//cout<<"Pz of the neutrino ="<<Solution<<endl;
	//cout<<"Pz of the gen-neutrino ="<<(double)TS.getGenLepn().pz()<<endl;
	//cout<<"Solution1 = "<<Solution1<<endl;
	//cout<<"Solution2 = "<<Solution2<<endl;

//Et-Sum of the lightest jets

	double EtSum = TopJets[2].getRecJet().et()+TopJets[3].getRecJet().et();
	//double HT    = TopJets[0].getRecJet().et()+TopJets[1].getRecJet().et()+TopJets[2].getRecJet().et()+TopJets[3].getRecJet().et();
	double Obs1 = (EtSum>0 ? EtSum : -1);
	evtselectVarVal.push_back(pair<unsigned int,double>(1,Obs1));
	
//Log of the difference in Bdisc between the 2nd and the 3rd jets (ordered in Bdisc)

	//sort the TopJets in Bdiscriminant
	std::sort(TopJets.begin(),TopJets.end(),BdiscComparator);
	
	double BGap = TopJets[1].getBdiscriminant() - TopJets[2].getBdiscriminant();
	double Obs2 = (BGap>0 ? log(BGap) : -1);
	evtselectVarVal.push_back(pair<unsigned int,double>(2,Obs2));
	
//Circularity of the event

	double N=0,D=0,C_tmp=0,C=1000;
	double nx,ny,nz;
	
	// C = 2min(E(pt.n)^2/E(pt)^2) = 2*N/D but it is theorically preferable to use C'=PI/2*min(E|pt.n|/E|pt|), sum over all jets+lepton+MET (cf PhysRevD 48 R3953(Nov 1993))
	
	for(unsigned int i=0;i<4;i++)
	{
		D += fabs(TopJets[i].getRecJet().pt());
	}
		D += fabs(TS.getRecLept().pt())+fabs(TS.getRecLepn().pt());

	if((D>0))
	{

		// Loop over all the unit vectors in the transverse plane in order to find the miminum : 
		for(unsigned int i=0; i<360; i++)
		{

			nx = cos((2*PI/360)*i);
			ny = sin((2*PI/360)*i);
			nz = 0;
			N=0;
			
			for(unsigned int i=0;i<4;i++)
			{
				N += fabs(TopJets[i].getRecJet().px()*nx+TopJets[i].getRecJet().py()*ny+TopJets[i].getRecJet().pz()*nz);
			}
				N += fabs(Lept->Px()*nx+Lept->Py()*ny+Lept->Pz()*nz)+fabs(Lepn->Px()*nx+Lepn->Py()*ny+Lepn->Pz()*nz);
	
			C_tmp = 2*N/D;
			if(C_tmp<C) C = C_tmp;
			
		}
	}
	
	double Obs3 = ( C!=1000 ? C : -1);
	evtselectVarVal.push_back(pair<unsigned int,double>(3,Obs3));

//HT variable (Et-sum of the four jets)

	double HT=0;
	for(unsigned int i=0;i<4;i++)
	{
		HT += TopJets[i].getRecJet().et();
	}
	
	double Obs4 = ( HT!=0 ? HT : -1);
	evtselectVarVal.push_back(pair<unsigned int,double>(4,Obs4));

//Transverse Mass of the system

	XYZTLorentzVector pjets;
	// for the four jets 
	for(unsigned int i=0;i<4;i++)
	{
		pjets += TopJets[i].getRecJet().p4();
	}
	// for the lepton
	pjets += TS.getRecLept().p4();
	// for the ~"neutrino"	
	double MET = TS.getMET().et();

	double MT = sqrt(pow(pjets.mass(),2)+pow(MET,2))+MET;
	
	double Obs5 = ( MT>0 ? MT : -1);
	evtselectVarVal.push_back(pair<unsigned int,double>(5,Obs5));

//CosTheta(Hadp,Hadq) 

	//sort the TopJets in Et
	std::sort(TopJets.begin(),TopJets.end(),EtComparator);
	
	double px1 = TopJets[2].getRecJet().px();     double px2 = TopJets[3].getRecJet().px();
	double py1 = TopJets[2].getRecJet().py();     double py2 = TopJets[3].getRecJet().py();
	double pz1 = TopJets[2].getRecJet().pz();     double pz2 = TopJets[3].getRecJet().pz();
	double E1  = TopJets[2].getRecJet().energy(); double E2 = TopJets[3].getRecJet().energy();
	//TLorentzVector *pjj = new TLorentzVector();
	//pjj->SetPxPyPzE(px1+px2,py1+py2,pz1+pz2,E1+E2);
	//TVector3 BoostBackToCM = -pjj->BoostVector();
	//pjj->Px()=0 if ppj back boosted ,checked!
	//pjj->Boost(BoostBackToCM);
	//cout<<pjj->Px()<<endl;
	TLorentzVector *LightJet1 = new TLorentzVector();
	LightJet1->SetPxPyPzE(px1,py1,pz1,E1);
	//LightJet1->Boost(BoostBackToCM);
	TLorentzVector *LightJet2 = new TLorentzVector();
	LightJet2->SetPxPyPzE(px2,py2,pz2,E2);
	//LightJet2->Boost(BoostBackToCM);
	
	//double DR = sqrt(pow(LightJet1->Eta()-LightJet2->Eta(),2)+pow(LightJet1->Phi()-LightJet2->Phi(),2));
	//double CosTheta = (LightJet1->X()*LightJet2->X()+LightJet1->Y()*LightJet2->Y()+LightJet1->Z()*LightJet2->Z())/(sqrt(pow(LightJet1->X(),2)+pow(LightJet1->Y(),2)+pow(LightJet1->Z(),2))*sqrt(pow(LightJet2->Y(),2)+pow(LightJet2->Y(),2)+pow(LightJet2->Z(),2)));
	double CosTheta = cos(LightJet2->Angle(LightJet1->Vect()));
	
	double Obs6 = ( -1<CosTheta ? CosTheta : -2);
	evtselectVarVal.push_back(pair<unsigned int,double>(6,Obs6));
	
	//delete pjj;
	delete LightJet1;
	delete LightJet2;

// try to find out more powerful observables related to the b-disc
	
	//sort the TopJets in Bdiscriminant
	std::sort(TopJets.begin(),TopJets.end(),BdiscComparator);
		
	double BjetsBdiscSum = TopJets[0].getBdiscriminant() + TopJets[1].getBdiscriminant();
	double LjetsBdiscSum = TopJets[2].getBdiscriminant() + TopJets[3].getBdiscriminant();
	
	//cout<<"BjetsBdiscSum = "<<BjetsBdiscSum<<endl;
	//cout<<"LjetsBdiscSum = "<<LjetsBdiscSum<<endl;
	
	double Obs7 = (LjetsBdiscSum !=0 ? log(BjetsBdiscSum/LjetsBdiscSum) : -1);
	evtselectVarVal.push_back(pair<unsigned int,double>(7,Obs7));
	
	double Obs8 = (BGap>0 ? log(BjetsBdiscSum*BGap) : -1);
	evtselectVarVal.push_back(pair<unsigned int,double>(8,Obs8));

// Missing transverse energy

	double Obs9 = (MET!=0 ? MET : -1);
	evtselectVarVal.push_back(pair<unsigned int,double>(9,Obs9));	

// Et-Ratio between the two highest in Et jets and four highest jets

	double Obs10 = (HT!=0 ? (HT-EtSum)/HT : -1);
	evtselectVarVal.push_back(pair<unsigned int,double>(10,Obs10));
	
// Pt of the lepton

	double Obs11 = TS.getRecLept().pt();
	evtselectVarVal.push_back(pair<unsigned int,double>(11,Obs11));
	
	
//Sphericity and Aplanarity without boosting back the system to CM frame

	TMatrixDSym Matrix(3);
	
	double PX2 = pow(Hadp->Px(),2)+pow(Hadq->Px(),2)+pow(Hadb->Px(),2)+pow(Lepb->Px(),2)+pow(Lept->Px(),2)+pow(Lepn->Px(),2);
	double PY2 = pow(Hadp->Py(),2)+pow(Hadq->Py(),2)+pow(Hadb->Py(),2)+pow(Lepb->Py(),2)+pow(Lept->Py(),2)+pow(Lepn->Py(),2);
	double PZ2 = pow(Hadp->Pz(),2)+pow(Hadq->Pz(),2)+pow(Hadb->Pz(),2)+pow(Lepb->Pz(),2)+pow(Lept->Pz(),2)+pow(Lepn->Pz(),2);
	
	double P2  = PX2+PY2+PZ2;
	
	double PXY = Hadp->Px()*Hadp->Py()+Hadq->Px()*Hadq->Py()+Hadb->Px()*Hadb->Py()+Lepb->Px()*Lepb->Py()+Lept->Px()*Lept->Py()+Lepn->Px()*Lepn->Py();
	double PXZ = Hadp->Px()*Hadp->Pz()+Hadq->Px()*Hadq->Pz()+Hadb->Px()*Hadb->Pz()+Lepb->Px()*Lepb->Pz()+Lept->Px()*Lept->Pz()+Lepn->Px()*Lepn->Pz();
	double PYZ = Hadp->Py()*Hadp->Pz()+Hadq->Py()*Hadq->Pz()+Hadb->Py()*Hadb->Pz()+Lepb->Py()*Lepb->Pz()+Lept->Py()*Lept->Pz()+Lepn->Py()*Lepn->Pz();

	Matrix(0,0) = PX2/P2; Matrix(0,1) = PXY/P2; Matrix(0,2) = PXZ/P2;
	Matrix(1,0) = PXY/P2; Matrix(1,1) = PY2/P2; Matrix(1,2) = PYZ/P2;
	Matrix(2,0) = PXZ/P2; Matrix(2,1) = PYZ/P2; Matrix(2,2) = PZ2/P2;
	
	TMatrixDSymEigen pTensor(Matrix);

	//if(fabs(pTensor.GetEigenValues().Sum()-1) > 0.01) cout<<"Sum of the eigen values not equal to 1!!!"<<endl;
	
	vector<double> EigValues;
	EigValues.clear();
	for(int i=0;i<3;i++)
	{
		EigValues.push_back(pTensor.GetEigenValues()[i]);
	}
	
	std::sort(EigValues.begin(),EigValues.end(),dComparator);
	
	double Sphericity = 1.5*(EigValues[1]+EigValues[2]);
	double Aplanarity = 1.5*EigValues[2];
	
	double Obs12 = (isnan(Sphericity) ? -1 : Sphericity);
	evtselectVarVal.push_back(pair<unsigned int,double>(12,Obs12));

	double Obs13 = (isnan(Aplanarity) ? -1 : Aplanarity);
	evtselectVarVal.push_back(pair<unsigned int,double>(13,Obs13));


//Sphericity and Aplanarity with boosting back the system to CM frame	

	TLorentzVector *TtbarSystem = new TLorentzVector();
	TtbarSystem->SetPx(Hadp->Px()+Hadq->Px()+Hadb->Px()+Lepb->Px()+Lept->Px()+Lepn->Px());
	TtbarSystem->SetPy(Hadp->Py()+Hadq->Py()+Hadb->Py()+Lepb->Py()+Lept->Py()+Lepn->Py());
	TtbarSystem->SetPz(Hadp->Pz()+Hadq->Pz()+Hadb->Pz()+Lepb->Pz()+Lept->Pz()+Lepn->Pz());
	TtbarSystem->SetE(Hadp->E()+Hadq->E()+Hadb->E()+Lepb->E()+Lept->E()+Lepn->E());

	TVector3 BoostBackToCM = -(TtbarSystem->BoostVector());
	Hadp->Boost(BoostBackToCM);
	Hadq->Boost(BoostBackToCM);
	Hadb->Boost(BoostBackToCM);
	Lepb->Boost(BoostBackToCM);
	Lept->Boost(BoostBackToCM);
	Lepn->Boost(BoostBackToCM);


	double BOOST_PX2 = pow(Hadp->Px(),2)+pow(Hadq->Px(),2)+pow(Hadb->Px(),2)+pow(Lepb->Px(),2)+pow(Lept->Px(),2)+pow(Lepn->Px(),2);
	double BOOST_PY2 = pow(Hadp->Py(),2)+pow(Hadq->Py(),2)+pow(Hadb->Py(),2)+pow(Lepb->Py(),2)+pow(Lept->Py(),2)+pow(Lepn->Py(),2);
	double BOOST_PZ2 = pow(Hadp->Pz(),2)+pow(Hadq->Pz(),2)+pow(Hadb->Pz(),2)+pow(Lepb->Pz(),2)+pow(Lept->Pz(),2)+pow(Lepn->Pz(),2);
	
	double BOOST_P2  = BOOST_PX2+BOOST_PY2+BOOST_PZ2;
	
	double BOOST_PXY = Hadp->Px()*Hadp->Py()+Hadq->Px()*Hadq->Py()+Hadb->Px()*Hadb->Py()+Lepb->Px()*Lepb->Py()+Lept->Px()*Lept->Py()+Lepn->Px()*Lepn->Py();
	double BOOST_PXZ = Hadp->Px()*Hadp->Pz()+Hadq->Px()*Hadq->Pz()+Hadb->Px()*Hadb->Pz()+Lepb->Px()*Lepb->Pz()+Lept->Px()*Lept->Pz()+Lepn->Px()*Lepn->Pz();
	double BOOST_PYZ = Hadp->Py()*Hadp->Pz()+Hadq->Py()*Hadq->Pz()+Hadb->Py()*Hadb->Pz()+Lepb->Py()*Lepb->Pz()+Lept->Py()*Lept->Pz()+Lepn->Py()*Lepn->Pz();

	TMatrixDSym BOOST_Matrix(3);

	BOOST_Matrix(0,0) = BOOST_PX2/BOOST_P2; BOOST_Matrix(0,1) = BOOST_PXY/BOOST_P2; BOOST_Matrix(0,2) = BOOST_PXZ/BOOST_P2;
	BOOST_Matrix(1,0) = BOOST_PXY/BOOST_P2; BOOST_Matrix(1,1) = BOOST_PY2/BOOST_P2; BOOST_Matrix(1,2) = BOOST_PYZ/BOOST_P2;
	BOOST_Matrix(2,0) = BOOST_PXZ/BOOST_P2; BOOST_Matrix(2,1) = BOOST_PYZ/BOOST_P2; BOOST_Matrix(2,2) = BOOST_PZ2/BOOST_P2;
	
	TMatrixDSymEigen BOOST_pTensor(BOOST_Matrix);

	//if(fabs(pTensor.GetEigenValues().Sum()-1) > 0.01) cout<<"Sum of the eigen values not equal to 1!!!"<<endl;
	
	vector<double> BOOST_EigValues;
	BOOST_EigValues.clear();
	for(int i=0;i<3;i++)
	{
		BOOST_EigValues.push_back(BOOST_pTensor.GetEigenValues()[i]);
	}
	
	std::sort(BOOST_EigValues.begin(),BOOST_EigValues.end(),dComparator);

	double BOOST_Sphericity = 1.5*(BOOST_EigValues[1]+BOOST_EigValues[2]);
	double BOOST_Aplanarity = 1.5*BOOST_EigValues[2];

	double Obs14 = ( isnan(BOOST_Sphericity) ? -1 : BOOST_Sphericity );
	evtselectVarVal.push_back(pair<unsigned int,double>(14,Obs14));

	double Obs15 = ( isnan(BOOST_Aplanarity) ? -1 : BOOST_Aplanarity );
	evtselectVarVal.push_back(pair<unsigned int,double>(15,Obs15));

// Centrality of the four jets

	double H=0;
	for(unsigned int i=0;i<4;i++)
	{
		H += TopJets[i].getRecJet().energy();
	}

	double Obs16 = ( H != 0 ? HT/H : -1 );
	evtselectVarVal.push_back(pair<unsigned int,double>(16,Obs16));

// distance from the origin in the (BjetsBdiscSum, LjetsBdiscSum)

	double Obs17 = ( BjetsBdiscSum != 0 && LjetsBdiscSum != 0 ? 0.707*(BjetsBdiscSum-LjetsBdiscSum) : -1 );
	evtselectVarVal.push_back(pair<unsigned int,double>(17,Obs17));

// Ratio of the Et Sum of the two lightest jets over HT=Sum of the Et of the four highest jets in Et

	double Obs18 = ( HT != 0 ? EtSum/HT : -1 );
	evtselectVarVal.push_back(pair<unsigned int,double>(18,Obs18));

// Put the vector in the TtSemiEvtSolution
	TS.setLRSignalEvtObservables(evtselectVarVal);

}

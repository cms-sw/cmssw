#include "L1Ntuple.h"
#include "hist.C"
#include "Style.C"
#include "iostream"
#include <vector>
#include <algorithm>
#include <utility>
#include <math.h>
using namespace std;
// --------------------------------------------------------------------
//                       MacroTemplate macro definition
// --------------------------------------------------------------------

class CompareGTGCT : public L1Ntuple
{
  public :

    //constructor    
    CompareGTGCT(std::string filename) : L1Ntuple(filename) {}
    CompareGTGCT() {}
    ~CompareGTGCT() {}

    //main function macro : arguments can be adpated to your need
    void run(Long64_t nevents);

  private : 

    //your private methods can be declared here
};


// --------------------------------------------------------------------
//                             run function 
// --------------------------------------------------------------------
void CompareGTGCT::run(Long64_t nevents)
{
  //load TDR style
  setTDRStyle();
  //number of events to process
  if (nevents==-1 || nevents>GetEntries()) nevents=GetEntries();
  std::cout << nevents << " to process ..." << std::endl;
  //loop over the events
  for (Long64_t i=0; i<nevents; i++)
    {
      //load the i-th event 
      Long64_t ientry = LoadTree(i); if (ientry < 0) break;
      GetEntry(i);

      //process progress
      if(i!=0 && (i%10000)==0) 
	std::cout << "- processing event " << i << "\r" << std::endl;

      //write your code here
int event_no = event_ -> event;
 //      Fill an array[128] containing the L1 trigger bits for the physics algorithm:       
	int PhysicsBits[128];
        for (int ibit=0; ibit < 128; ibit++) {
                 PhysicsBits[ibit] = 0;
                         if (ibit<64) {
                             PhysicsBits[ibit] = (gt_->tw1[2]>>ibit)&1;
                         }
                        else {
                            PhysicsBits[ibit] = (gt_->tw2[2]>>(ibit-64))&1;
                        }
                      }
//////////////////////Jets/////////////////////////
	int Njet = gt_ -> Njet;

	vector<float> pt_val, eta_val, phi_val;
	vector<int> bx_gt;
	for(int ujet=0; ujet <  Njet; ujet++){
		int bx = gt_ -> Bxjet[ujet];
		//if(bx != 0) continue;
		bx_gt.push_back(bx);

		float rank = gt_ -> Rankjet[ujet];
		float pt = rank*4.;
    		pt_val.push_back(pt);	
		float eta = gt_ -> Etajet[ujet];
		eta_val.push_back(eta);
		float phi = gt_ -> Phijet[ujet];
		phi_val.push_back(phi);
    	}
	bool problem = false;//switch to true for printout
	if(problem){//gt printout
  	cout<<event_no<<endl;
	cout<<"Problem at event: "<<event_no<<endl;
	cout<<"N of gt jets = "<<Njet<<endl;
	cout<<"Jet# "<<"\t"<<"bx"<<"\t"<<"pt"<<"\t"<<"eta"<<"\t"<<"phi"<<endl;
	for(int iii=0; iii<Njet; iii++){
	cout<<iii<<"\t"<<bx_gt[iii]<<"\t"<<pt_val[iii]<<"\t"<<eta_val[iii]<<"\t"<<phi_val[iii]<<endl;
	}
	}

///////end of gt

/////gct
//int size_c = gct_->CJetSize[int.i];
int total_gct_jets=0;
vector<float> pt_gct_c, eta_gct_c, phi_gct_c;
vector<int> bx_gct_c;
for(int cjet=0; cjet<20; cjet++){
		int bx_gct_c_c = gct_ -> CJetBx[cjet];
		bx_gct_c.push_back(bx_gct_c_c);
		//if(bx_gct_c != 0) continue;
		float rank_gct_c = gct_->CJetRnk[cjet]; if (rank_gct_c>0 && ((abs(bx_gct_c_c) % 4) != 2) ) total_gct_jets++;
		float pt_c = rank_gct_c * 4.;
		pt_gct_c.push_back(pt_c);
		float phi_c = gct_->CJetPhi[cjet];
		phi_gct_c.push_back(phi_c);
		float eta_c = gct_->CJetEta[cjet];

		eta_gct_c.push_back(eta_c);
}
	
//int size_f = gct_->FJetSize[i];
vector<float> pt_gct_f, eta_gct_f, phi_gct_f;
vector<int> bx_gct_f;
for(int fjet=0; fjet<20; fjet++){
		int bx_gct_f_f = gct_ -> FJetBx[fjet];
		bx_gct_f.push_back(bx_gct_f_f);

		//if(bx_gct_f != 0) continue;
		float rank_gct_f = gct_->FJetRnk[fjet]; if (rank_gct_f>0 && ((abs(bx_gct_f_f) % 4) != 2) ) total_gct_jets++;
		float pt_f = rank_gct_f * 4.;
		pt_gct_f.push_back(pt_f);
                float phi_f = gct_->FJetPhi[fjet];
		phi_gct_f.push_back(phi_f);
                float eta_f = gct_->FJetEta[fjet];
		eta_gct_f.push_back(eta_f);

}
	
//int size_t = gct_->TJetSize[i];
vector<float> pt_gct_t, eta_gct_t, phi_gct_t;
vector<int> bx_gct_t;
for(int tjet=0; tjet<20; tjet++){
		int bx_gct_t_t = gct_ -> TJetBx[tjet];
		bx_gct_t.push_back(bx_gct_t_t);

		//if(bx_gct_t !=0) continue;
		float rank_gct_t = gct_->TJetRnk[tjet]; if (rank_gct_t>0 && ((abs(bx_gct_t_t) % 4) != 2) ) total_gct_jets++;
		float pt_t = rank_gct_t * 4.;
		pt_gct_t.push_back(pt_t);
                float phi_t = gct_->TJetPhi[tjet];
		phi_gct_t.push_back(phi_t);
                float eta_t = gct_->TJetEta[tjet];
		eta_gct_t.push_back(eta_t);


}
if(problem){//gct printout

cout<<"Jet #"<<"\t"<<"ID"<<"\t"<<"bx"<<"\t"<<"pt"<<"\t\t"<<"eta"<<"\t\t"<<"phi"<<endl;

for(int j=0; j<20; j++){

cout<<j<<"\t"<<"C"<<"\t"<<bx_gct_c[j]<<"\t"<<pt_gct_c[j]<<"\t\t"<<eta_gct_c[j]<<"\t\t"<<phi_gct_c[j]<<endl;
cout<<j<<"\t"<<"T"<<"\t"<<bx_gct_t[j]<<"\t"<<pt_gct_t[j]<<"\t\t"<<eta_gct_t[j]<<"\t\t"<<phi_gct_t[j]<<endl;
cout<<j<<"\t"<<"F"<<"\t"<<bx_gct_f[j]<<"\t"<<pt_gct_f[j]<<"\t\t"<<eta_gct_f[j]<<"\t\t"<<phi_gct_f[j]<<endl;

}

}

////end of gct

/////////matching gt to gct////
int match=0;
if(Njet>0){
for(int k=0; k<Njet; k++){
        for(int j=0; j<20; j++){
        if((bx_gt[k]==bx_gct_c[j]) && (pt_val[k]==pt_gct_c[j]) && (eta_val[k]==eta_gct_c[j]) && (phi_val[k]==phi_gct_c[j])) {match++; }
        if((bx_gt[k]==bx_gct_f[j]) && (pt_val[k]==pt_gct_f[j]) && (eta_val[k]==eta_gct_f[j]) && (phi_val[k]==phi_gct_f[j])) {match++; }
        if((bx_gt[k]==bx_gct_t[j]) && (pt_val[k]==pt_gct_t[j]) && (eta_val[k]==eta_gct_t[j]) && (phi_val[k]==phi_gct_t[j])) {match++; }
	
}
}
}///end of matching
if(match != Njet) std::cout<<"No match - GT information not found in GCT   "<<i<<"   "<<Njet<<"  "<<match<<std::endl;

int match2=0;
///matching gct to gt////
if(total_gct_jets>0){
for(int j=0; j<20; j++){
        for(int k=0; k<Njet; k++){
        if(pt_gct_c[j]>0 && (bx_gt[k]==bx_gct_c[j]) && (pt_val[k]==pt_gct_c[j]) && (eta_val[k]==eta_gct_c[j]) && (phi_val[k]==phi_gct_c[j])) {match2++;}
        if(pt_gct_f[j]>0 && (bx_gt[k]==bx_gct_f[j]) && (pt_val[k]==pt_gct_f[j]) && (eta_val[k]==eta_gct_f[j]) && (phi_val[k]==phi_gct_f[j])) {match2++;}
        if(pt_gct_t[j]>0 && (bx_gt[k]==bx_gct_t[j]) && (pt_val[k]==pt_gct_t[j]) && (eta_val[k]==eta_gct_t[j]) && (phi_val[k]==phi_gct_t[j])) {match2++;}

}
}
}///end of matching
if(match2 != total_gct_jets) std::cout<<"No match - GCT information not found in GT   "<<i<<"   "<<total_gct_jets<<"   "<<match2<<std::endl;


////end of matching


}//end of events

}

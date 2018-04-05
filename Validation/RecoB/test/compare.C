#include <vector>
#include <string>
using namespace std;

TGraphErrors * computeGraph(TH1F * effdiscrb, TH1F* effdiscruds);

int colorList[] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20}; 
int markerStyleList[] = {21,21,23,23,22,22,23,23,24,24,24,24,25,25,25,25,26,26,26,26};  

TObject * getHistogram(TFile * f, string algo,string histoName, string range, string suffix)
{
string prefix = "DQMData/Run 1/Btag/Run summary/JetTag";
string d = prefix+"_"+algo+"_"+range;
 cout <<" STR: "<<d<<endl; 
 cout <<" DIR: "<<  f->Get(d.c_str())<<endl;
 cout <<" HIS: "<<  (histoName+"_"+algo+"_"+range+suffix).c_str()<<endl;

 TDirectory * dir  =(TDirectory *) f->Get(d.c_str());
 return dir->Get((histoName+"_"+algo+"_"+range+suffix).c_str());
}

TObject * getHistogram2(TFile * f, string algo,string histoName, string range, string suffix)
{
string prefix = "JetTag";
string d = prefix+"_"+algo+"_"+range;
 cout <<" STR: "<<d<<endl; 
 cout <<" DIR: "<<  f->Get(d.c_str())<<endl;
 cout <<" HIS: "<<  (histoName+"_"+algo+"_"+range+suffix).c_str()<<endl;

 TDirectory * dir  =(TDirectory *) f->Get(d.c_str());
 return dir->Get((histoName+"_"+algo+"_"+range+suffix).c_str());
}




TObject * getHistogram2(TFile * f, string algo,string histoName, string range = "GLOBAL")
{
string prefix = "JetTag";
string d = prefix+"_"+algo+"_"+range;
TDirectory * dir  =(TDirectory *) f->Get(d.c_str());
return dir->Get((histoName+"_"+algo+"_"+range).c_str());
}

void setStyle(int i, TGraph *obj)
{
obj->SetMarkerColor(colorList[i]);
obj->SetLineColor(colorList[i]);
obj->SetMarkerStyle(markerStyleList[i]);
obj->SetMarkerSize(.8);
}



TGraphErrors *  drawAll()
{
  TFile *_file1 = TFile::Open("here.root");

  // use also an old style file??

  bool old = false;
  TFile *_file2;
  if (old == true){
  _file2= TFile::Open("jetTagAnalysis.root");
  }
  vector<TFile *> files;
  vector<string> algos;
  algos.push_back("trackCountingHighPurBJetTags");
  algos.push_back("trackCountingHighEffBJetTags");
  algos.push_back("jetProbabilityBJetTags");
  algos.push_back("jetBProbabilityBJetTags");
  algos.push_back("simpleSecondaryVertexBJetTags");
  algos.push_back("combinedSecondaryVertexBJetTags");
  algos.push_back("softMuonBJetTags");
  algos.push_back("softMuonByIP3dBJetTags");
  algos.push_back("softMuonByPtBJetTags");
  algos.push_back("softElectronBJetTags");

  TLegend * leg = new TLegend(0.2,0.65,0.4,0.9);
  TCanvas * c1 = new TCanvas();
  c1->SetLogy();  
  c1->SetGridy();  
  c1->SetGridx();  
  for(int i = 0 ; i < algos.size() ; i++)
   {
      cout << algos[i] << endl;
      // get the eff vs dicriminant

      TH1F* effdiscrb =  (TH1F *) getHistogram(_file1,algos[i],"effVsDiscrCut_discr","GLOBAL", "B");
      TH1F* effdiscruds = (TH1F *) getHistogram(_file1,algos[i],"effVsDiscrCut_discr","GLOBAL", "DUS");
      
      cout <<" HISTOS "<<effdiscrb <<" " <<effdiscruds<<endl;

      TGraphErrors * h = computeGraph(effdiscrb,effdiscruds);
      //      h->SetMaximum(1.);
            h->SetMinimum(1e-5);
      //      h->GetXaxis()->SetLimits(0,.2);
      //      h->GetYaxis()->SetLimits(1e-4,1);
     cout << h << endl;
     //     return h;
     setStyle(i,h);
     if(i==0) h->Draw("ALP"); else h->Draw("lpsame"); 
     leg->AddEntry(h,algos[i].c_str(),"p");
   }

  if (old == true){
    
    for(int i = 0 ; i < algos.size() ; i++)
      {
	cout << algos[i] << endl;
	// get the eff vs dicriminant
	
	TH1F* effdiscrb =  (TH1F *) getHistogram2(_file2,algos[i],"effVsDiscrCut_discr","GLOBAL", "B");
	TH1F* effdiscruds = (TH1F *) getHistogram2(_file2,algos[i],"effVsDiscrCut_discr","GLOBAL", "DUS");
      
	cout <<" HISTOS "<<effdiscrb <<" " <<effdiscruds<<endl;
	
	TGraphErrors * h = computeGraph(effdiscrb,effdiscruds);
	//      h->SetMaximum(1.);
	h->SetMinimum(1e-5);
	//      h->GetXaxis()->SetLimits(0,.2);
	//      h->GetYaxis()->SetLimits(1e-4,1);
	cout << h << endl;
	//     return h;
     setStyle(i+10,h);
     //     if(i==0) h->Draw("ALP"); else 
     h->Draw("lpsame"); 
     leg->AddEntry(h,algos[i].c_str(),"p");
      }
    
  }



  leg->Draw("same");

  return 0;
}


TGraphErrors * computeGraph(TH1F * effdiscrb, TH1F* effdiscruds){
  double be[1000],ber[1000], udse[1000], udser[1000];
  int nbins = effdiscrb->GetNbinsX();
  cout <<" BINS = "<<nbins<<endl;
  for (int i=0; i<nbins; i++)    {
    be[i] = effdiscrb->GetBinContent(i+1);
    ber[i] = effdiscrb->GetBinError(i+1);
    udse[i] = effdiscruds->GetBinContent(i+1);
    udser[i] = effdiscruds->GetBinError(i+1);
    cout <<" GOT "<< i<<" " << be[i]<<" " <<ber[i]<<" " <<udse[i]<<" " <<udser[i]<<endl;
  }
  TGraphErrors * result = new TGraphErrors(nbins, be, udse, ber, udser);
  //  result->Draw();
  return result;
}



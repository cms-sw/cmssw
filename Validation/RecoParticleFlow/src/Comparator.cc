#include <TFile.h>
#include <TH1.h>
#include <TH2.h>
#include <TStyle.h>
#include <TPaveStats.h>


#include <sstream>

#include "Validation/RecoParticleFlow/interface/Comparator.h"
#include "Validation/RecoParticleFlow/interface/TH2Analyzer.h"
#include "Validation/RecoParticleFlow/interface/NicePlot.h"

using namespace std; 

void Comparator::SetDirs( const char* file0,
			  const char* dir0,
			  const char* file1,
			  const char* dir1  ) {
  
  file0_ = new TFile( file0 );
  if( file0_->IsZombie() ) exit(1);
  dir0_ = file0_->GetDirectory( dir0 );
  if(! dir0_ ) exit(1);
  
  file1_ = new TFile( file1 );
  if( file1_->IsZombie() ) exit(1);
  dir1_ = file1_->GetDirectory( dir1 );
  if(! dir1_ ) exit(1);
}


void Comparator::SetStyles( Style* s0, 
		  Style* s1,
		  const char* leg0,
		  const char* leg1) { 
  s0_ = s0; 
  s1_ = s1;
  
  legend_.Clear();
  legend_.AddEntry( s0_, leg0, "mlf");
  legend_.AddEntry( s1_, leg1, "mlf");
}
  
void Comparator::DrawSlice( const char* key, 
			    int binxmin, int binxmax, 
			    Mode mode ) {
    
  static int num = 0;
    
  ostringstream out0;
  out0<<"h0_2d_"<<num;
  ostringstream out1;
  out1<<"h1_2d_"<<num;
  num++;

  string name0 = out0.str();
  string name1 = out1.str();
      

  TH1* h0 = Histo( key, 0);
  TH1* h1 = Histo( key, 1);

  TH2* h0_2d = dynamic_cast< TH2* >(h0);
  TH2* h1_2d = dynamic_cast< TH2* >(h1);
    
  if(h0_2d->GetNbinsY() == 1 || 
     h1_2d->GetNbinsY() == 1 ) {
    cerr<<key<<" is not 2D"<<endl;
    return;
  }
    
  TH1::AddDirectory( false );

  TH1D* h0_slice = h0_2d->ProjectionY(name0.c_str(),
				      binxmin, binxmax, "");
  TH1D* h1_slice = h1_2d->ProjectionY(name1.c_str(),
				      binxmin, binxmax, "");
  TH1::AddDirectory( true );
  Draw( h0_slice, h1_slice, mode);        
}



//void Comparator::DrawMeanSlice(const char* key, const unsigned int binxmin, const unsigned int binxmax,
//			       const unsigned int nbin, const double Ymin, const double Ymax,
//			       const std::string title, const std::string binning_option)
//{
//  TDirectory* dir = dir1_;
//  dir->cd();
//  //gStyle->SetOptStat("");
//  TH2Analyzer *h2 = (TH2Analyzer*) dir->Get(key);
//  TH1D* meanslice = new TH1D("meanslice","MeanSlice",nbin,binxmin,binxmax);
//  h2->MeanSlice(meanslice,binxmin,binxmax,nbin,binning_option);
//  meanslice->SetTitle(title.c_str());
//  meanslice->SetMaximum(Ymax);
//  meanslice->SetMinimum(Ymin);
//  meanslice->SetMarkerStyle(21);
//  meanslice->SetMarkerColor(4);
//  meanslice->Draw("E1");
//
//  dir = dir0_;
//  dir->cd();
//  TH2Analyzer *h2b = (TH2Analyzer*) dir->Get(key);
//  TH1D* meanslice2 = new TH1D("meanslice2","MeanSlice",nbin,binxmin,binxmax);
//  h2b->MeanSlice(meanslice2,binxmin,binxmax,nbin,binning_option);
//  meanslice2->SetTitle(title.c_str());
//  meanslice2->SetMaximum(Ymax);
//  meanslice2->SetMinimum(Ymin);
//  meanslice2->SetMarkerStyle(21);
//  meanslice2->SetMarkerColor(2);
//  meanslice2->Draw("E1same");
//}
//
//void Comparator::DrawSigmaSlice(const char* key, const unsigned int binxmin, const unsigned int binxmax,
//				const unsigned int nbin, const double Ymin, const double Ymax,
//				const std::string title, const std::string binning_option)
//{
//  TDirectory* dir = dir1_;
//  dir->cd();
//  gStyle->SetOptStat("");
//  TH2Analyzer *h2 = (TH2Analyzer*) dir->Get(key);
//  TH1D* sigmaslice = new TH1D("sigmaslice","Sigmaslice",nbin,binxmin,binxmax);
//  h2->SigmaSlice(sigmaslice,binxmin,binxmax,nbin,binning_option);
//  sigmaslice->SetTitle(title.c_str());
//  sigmaslice->SetMaximum(Ymax);
//  sigmaslice->SetMinimum(Ymin);
//  sigmaslice->SetMarkerStyle(21);
//  sigmaslice->SetMarkerColor(4);
//  sigmaslice->Draw("E1");
//
//  dir = dir0_;
//  dir->cd();
//  TH2Analyzer *h2b = (TH2Analyzer*) dir->Get(key);
//  TH1D* sigmaslice2 = new TH1D("sigmaslice2","Sigmaslice",nbin,binxmin,binxmax);
//  h2b->SigmaSlice(sigmaslice2,binxmin,binxmax,nbin,binning_option);
//  sigmaslice2->SetTitle(title.c_str());
//  sigmaslice2->SetMaximum(Ymax);
//  sigmaslice2->SetMinimum(Ymin);
//  sigmaslice2->SetMarkerStyle(21);
//  sigmaslice2->SetMarkerColor(2);
//  sigmaslice2->Draw("E1same");
//}
//
//void Comparator::DrawGaussSigmaSlice(const char* key, const unsigned int binxmin, const unsigned int binxmax,
//				     const unsigned int nbin, const double Ymin, const double Ymax,
//				     const std::string title, const std::string binning_option, const unsigned int rebin,
//				     const double fitmin, const double fitmax, const std::string epsname,
//				     const bool doFit)
//{
//  TDirectory* dir = dir1_;
//  dir->cd();
//  gStyle->SetOptStat("");
//  TH2Analyzer *h2 = (TH2Analyzer*) dir->Get(key);
//  TH1D* sigmaslice = new TH1D("sigmaslice","Sigmaslice",nbin,binxmin,binxmax);
//  h2->SigmaSlice(sigmaslice,binxmin,binxmax,nbin,binning_option);
////   TF1 *fitfcndgssrms3 = new TF1("fitfcndgssrms3",(void*)Comparator::fitFunction_f,binxmin,binxmax,4);
////   if (doFit)
////     {
////       fitfcndgssrms3->SetNpx(500);
////       fitfcndgssrms3->SetLineWidth(3);
////       fitfcndgssrms3->SetLineStyle(2);
////       fitfcndgssrms3->SetLineColor(4);
////       sigmaslice->Fit("fitfcndgssrms3","0R");
////       //sigmaslice->Draw("E1");
////     }
//
//  TH1D* sigmasliceGauss = new TH1D("sigmasliceGauss","Sigmaslice",nbin,binxmin,binxmax);
//  h2->SigmaGaussSlice(sigmasliceGauss,binxmin,binxmax,nbin,binning_option,rebin,fitmin,fitmax,epsname);
//  sigmasliceGauss->SetTitle(title.c_str());
//  sigmasliceGauss->SetMaximum(Ymax);
//  sigmasliceGauss->SetMinimum(Ymin);
//  sigmasliceGauss->SetMarkerStyle(21);
//  sigmasliceGauss->SetMarkerColor(4);
//
////   TF1 *fitfcndgsse3 = new TF1("fitfcndgsse3",(void*)(Comparator::fitFunction_f),binxmin,binxmax,4);
////   if (doFit)
////     {
////       fitfcndgsse3->SetNpx(500);
////       fitfcndgsse3->SetLineWidth(3);
////       fitfcndgsse3->SetLineStyle(1);
////       fitfcndgsse3->SetLineColor(4);
////       sigmasliceGauss->Fit("fitfcndgsse3","0R");
////     }
//
//  dir = dir0_;
//  dir->cd();
//  TH2Analyzer *h2b = (TH2Analyzer*) dir->Get(key);
//  TH1D* sigmasliceb = new TH1D("sigmasliceb","Sigmasliceb",nbin,binxmin,binxmax);
//  h2b->SigmaSlice(sigmasliceb,binxmin,binxmax,nbin,binning_option);
////   TF1 *fitfcndgssrmsb3 = new TF1("fitfcndgssrmsb3",(void*)(Comparator::fitFunction_f),binxmin,binxmax,4);
////   if (doFit)
////     {
////       fitfcndgssrmsb3->SetNpx(500);
////       fitfcndgssrmsb3->SetLineWidth(3);
////       fitfcndgssrmsb3->SetLineStyle(2);
////       fitfcndgssrmsb3->SetLineColor(2);
////       sigmasliceb->Fit("fitfcndgssrmsb3","0R");
////     }
//
//  TH1D* sigmaslicebGauss = new TH1D("sigmaslicebGauss","Sigmasliceb",nbin,binxmin,binxmax);
//  h2b->SigmaGaussSlice(sigmaslicebGauss,binxmin,binxmax,nbin,binning_option,rebin,fitmin,fitmax,epsname+"b");
//  sigmaslicebGauss->SetTitle(title.c_str());
//  sigmaslicebGauss->SetMaximum(Ymax);
//  sigmaslicebGauss->SetMinimum(Ymin);
//  sigmaslicebGauss->SetMarkerStyle(21);
//  sigmaslicebGauss->SetMarkerColor(2);
//
////   TF1 *fitfcndgsseb3 = new TF1("fitfcndgsseb3",(void*)(Comparator::fitFunction_f),binxmin,binxmax,4);
////   if (doFit)
////     {
////       fitfcndgsseb3->SetNpx(500);
////       fitfcndgsseb3->SetLineWidth(3);
////       fitfcndgsseb3->SetLineStyle(1);
////       fitfcndgsseb3->SetLineColor(2);
////       sigmaslicebGauss->Fit("fitfcndgsseb3","0R");
////     }
////   sigmasliceGauss->Draw("E1");
////   if (doFit)
////     {
////       fitfcndgssrms3->DrawClone("same"); 
////       fitfcndgsse3->DrawClone("same"); 
////     }
////   sigmaslicebGauss->Draw("E1same");
////   if (doFit)
////     {
////       fitfcndgssrmsb3->DrawClone("same"); 
////       fitfcndgsseb3->DrawClone("same"); 
////     }
//}
//
//void Comparator::DrawGaussSigmaOverMeanXSlice(const char* key, const unsigned int binxmin, const unsigned int binxmax,
//					      const unsigned int nbin, const double Ymin, const double Ymax,
//					      const std::string title, const std::string binning_option, const unsigned int rebin,
//					      const double fitmin, const double fitmax, const std::string epsname)
//{
//  TDirectory* dir = dir1_;
//  dir->cd();
//  gStyle->SetOptStat("");
//  TH2Analyzer *h2 = (TH2Analyzer*) dir->Get(key);
//  TH1D* sigmaslice = new TH1D("sigmaslice","Sigmaslice",nbin,binxmin,binxmax);
//  h2->SigmaSlice(sigmaslice,binxmin,binxmax,nbin,binning_option);
//
//  TH1D* meanXslice = new TH1D("meanXslice","MeanXslice",nbin,binxmin,binxmax);
//  h2->MeanXSlice(meanXslice,binxmin,binxmax,nbin,binning_option);
//  //meanXslice->Draw("E1");
//  sigmaslice->Divide(meanXslice);
//  //sigmaslice->Draw("E1");
//
////   TF1 *fitFcnrms3 = new TF1("fitFcnrms3",(void*)(Comparator::fitFunction_f),binxmin,binxmax,4);
////   fitFcnrms3->SetNpx(500);
////   fitFcnrms3->SetLineWidth(3);
////   fitFcnrms3->SetLineStyle(2);
////   fitFcnrms3->SetLineColor(4);
////   sigmaslice->Fit("fitFcnrms3","0R");
//
//  TH1D* sigmasliceGauss = new TH1D("sigmasliceGauss","Sigmaslice",nbin,binxmin,binxmax);
//  h2->SigmaGaussSlice(sigmasliceGauss,binxmin,binxmax,nbin,binning_option,rebin,fitmin,fitmax,epsname);
//  sigmasliceGauss->Divide(meanXslice);
//  sigmasliceGauss->SetTitle(title.c_str());
//  sigmasliceGauss->SetMaximum(Ymax);
//  sigmasliceGauss->SetMinimum(Ymin);
//  sigmasliceGauss->SetMarkerStyle(21);
//  sigmasliceGauss->SetMarkerColor(4);
//
////   TF1 *fitFcne3 = new TF1("fitFcne3",(void*)(Comparator::fitFunction_f),binxmin,binxmax,4);
////   fitFcne3->SetNpx(500);
////   fitFcne3->SetLineWidth(3);
////   fitFcne3->SetLineStyle(1);
////   fitFcne3->SetLineColor(4);
////   sigmasliceGauss->Fit("fitFcne3","0R");
//
//  dir = dir0_;
//  dir->cd();
//  TH2Analyzer *h2b = (TH2Analyzer*) dir->Get(key);
//  TH1D* sigmasliceb = new TH1D("sigmasliceb","Sigmasliceb",nbin,binxmin,binxmax);
//  h2b->SigmaSlice(sigmasliceb,binxmin,binxmax,nbin,binning_option);
//  sigmasliceb->Divide(meanXslice);
//
////   TF1 *fitFcnrmsb3 = new TF1("fitFcnrmsb3",(void*)(Comparator::fitFunction_f),binxmin,binxmax,4);
////   fitFcnrmsb3->SetNpx(500);
////   fitFcnrmsb3->SetLineWidth(3);
////   fitFcnrmsb3->SetLineStyle(2);
////   fitFcnrmsb3->SetLineColor(2);
////   sigmasliceb->Fit("fitFcnrmsb3","0R");
//
//  TH1D* sigmaslicebGauss = new TH1D("sigmaslicebGauss","Sigmasliceb",nbin,binxmin,binxmax);
//  h2b->SigmaGaussSlice(sigmaslicebGauss,binxmin,binxmax,nbin,binning_option,rebin,fitmin,fitmax,epsname+"b");
//  sigmaslicebGauss->Divide(meanXslice);
//  sigmaslicebGauss->SetTitle(title.c_str());
//  sigmaslicebGauss->SetMaximum(Ymax);
//  sigmaslicebGauss->SetMinimum(Ymin);
//  sigmaslicebGauss->SetMarkerStyle(21);
//  sigmaslicebGauss->SetMarkerColor(2);
//
////   TF1 *fitFcneb3 = new TF1("fitFcneb3",(void*)(Comparator::fitFunction_f),binxmin,binxmax,4);
////   fitFcneb3->SetNpx(500);
////   fitFcneb3->SetLineWidth(3);
////   fitFcneb3->SetLineStyle(1);
////   fitFcneb3->SetLineColor(2);
////   sigmaslicebGauss->Fit("fitFcneb3","0R");
//
//  sigmasliceGauss->Draw("E1");
////   fitFcnrms3->DrawClone("same"); 
////   fitFcne3->DrawClone("same"); 
//  sigmaslicebGauss->Draw("E1same");
////   fitFcnrmsb3->DrawClone("same"); 
////   fitFcneb3->DrawClone("same"); 
//}
//
//void Comparator::DrawGaussSigmaOverMeanSlice(const char* key, const char* key2, const unsigned int binxmin, const unsigned int binxmax,
//					     const unsigned int nbin, const double Ymin, const double Ymax,
//					     const std::string title, const std::string binning_option, const unsigned int rebin,
//					     const double fitmin, const double fitmax, const std::string epsname)
//{
//  TDirectory* dir = dir1_;
//  dir->cd();
//  gStyle->SetOptStat("");
//  TH2Analyzer *h2 = (TH2Analyzer*) dir->Get(key);
//  //TH1D* sigmaslice = new TH1D("sigmaslice","Sigmaslice",nbin,binxmin,binxmax);
//  //h2->SigmaSlice(sigmaslice,binxmin,binxmax,nbin,binning_option);
//
//  TH2Analyzer *h2_b = (TH2Analyzer*) dir->Get(key2);
//  TH1D* meanslice = new TH1D("meanslice","Meanslice",nbin,binxmin,binxmax);
//  h2_b->MeanSlice(meanslice,binxmin,binxmax,nbin,binning_option);
//  //sigmaslice->Divide(meanslice);
//  //sigmaslice->Draw("E1");
//
//  //TF1 *fitFcnrms3 = new TF1("fitFcnrms3",(void*)(Comparator::fitFunction_f),binxmin,binxmax,4);
//  //fitFcnrms3->SetNpx(500);
//  //fitFcnrms3->SetLineWidth(3);
//  //fitFcnrms3->SetLineStyle(2);
//  //fitFcnrms3->SetLineColor(4);
//  //sigmaslice->Fit("fitFcnrms3","0R");
//
//  TH1D* sigmasliceGauss = new TH1D("sigmasliceGauss","Sigmaslice",nbin,binxmin,binxmax);
//  h2->SigmaGaussSlice(sigmasliceGauss,binxmin,binxmax,nbin,binning_option,rebin,fitmin,fitmax,epsname);
//  sigmasliceGauss->Divide(meanslice);
//  sigmasliceGauss->SetTitle(title.c_str());
//  sigmasliceGauss->SetMaximum(Ymax);
//  sigmasliceGauss->SetMinimum(Ymin);
//  sigmasliceGauss->SetMarkerStyle(21);
//  sigmasliceGauss->SetMarkerColor(4);
//
//  //TF1 *fitFcne3 = new TF1("fitFcne3",(void*)(Comparator::fitFunction_f),binxmin,binxmax,4);
//  //fitFcne3->SetNpx(500);
//  //fitFcne3->SetLineWidth(3);
//  //fitFcne3->SetLineStyle(1);
//  //fitFcne3->SetLineColor(4);
//  //sigmasliceGauss->Fit("fitFcne3","0R");
//
//  dir = dir0_;
//  dir->cd();
//  TH2Analyzer *h2b = (TH2Analyzer*) dir->Get(key);
//  TH2Analyzer *h2b_b = (TH2Analyzer*) dir->Get(key2);
//  TH1D* meansliceb = new TH1D("meansliceb","Meanslice",nbin,binxmin,binxmax);
//  h2b_b->MeanSlice(meansliceb,binxmin,binxmax,nbin,binning_option);
//
//  //TH1D* sigmasliceb = new TH1D("sigmasliceb","Sigmasliceb",nbin,binxmin,binxmax);
//  // h2b->SigmaSlice(sigmasliceb,binxmin,binxmax,nbin,binning_option);
//  //sigmasliceb->Divide(meansliceb);
//
//  //TF1 *fitFcnrmsb3 = new TF1("fitFcnrmsb3",(void*)(Comparator::fitFunction_f),binxmin,binxmax,4);
//  //fitFcnrmsb3->SetNpx(500);
//  //fitFcnrmsb3->SetLineWidth(3);
//  //fitFcnrmsb3->SetLineStyle(2);
//  //fitFcnrmsb3->SetLineColor(2);
//  //sigmasliceb->Fit("fitFcnrmsb3","0R");
//
//  TH1D* sigmaslicebGauss = new TH1D("sigmaslicebGauss","Sigmasliceb",nbin,binxmin,binxmax);
//  h2b->SigmaGaussSlice(sigmaslicebGauss,binxmin,binxmax,nbin,binning_option,rebin,fitmin,fitmax,epsname+"b");
//  sigmaslicebGauss->Divide(meansliceb);
//  sigmaslicebGauss->SetTitle(title.c_str());
//  sigmaslicebGauss->SetMaximum(Ymax);
//  sigmaslicebGauss->SetMinimum(Ymin);
//  sigmaslicebGauss->SetMarkerStyle(21);
//  sigmaslicebGauss->SetMarkerColor(2);
//
//  //TF1 *fitFcneb3 = new TF1("fitFcneb3",(void*)(Comparator::fitFunction_f),binxmin,binxmax,4);
//  //fitFcneb3->SetNpx(500);
//  //fitFcneb3->SetLineWidth(3);
//  //fitFcneb3->SetLineStyle(1);
//  //fitFcneb3->SetLineColor(2);
//  //sigmaslicebGauss->Fit("fitFcneb3","0R");
//
//  //meanslice->Draw("E1");
//  sigmasliceGauss->Draw("E1");
//  //fitFcnrms3->DrawClone("same"); 
//  //fitFcne3->DrawClone("same"); 
//  sigmaslicebGauss->Draw("E1same");
//  //fitFcnrmsb3->DrawClone("same"); 
//  //fitFcneb3->DrawClone("same"); 
//}
//
////COLIN mode is not used
//// loosing the possibility to pass options from TH1::Draw
//// remove, and use Comparator::Histo to access histograms from outside
//void Comparator::Draw2D_file1( const char* key, Mode mode) {
//  TDirectory* dir = dir0_;
//  dir->cd();
//  gStyle->SetPalette(1);
//  TH2F *h2 = (TH2F*) dir->Get(key);
//  h2->Draw("colz");
//}
//
////COLIN mode is not used
//// loosing the possibility to pass options from TH1::Draw
//// remove, and use Comparator::Histo to access histograms from outside
//// btw: file1 and file2 should not be hardcoded in the name of the function
//// but provided as an integer as for the other functions of the class
//void Comparator::Draw2D_file2( const char* key, Mode mode) {
//  TDirectory* dir = dir1_;
//  dir->cd();
//  gStyle->SetPalette(1);
//  TH2F *h2 = (TH2F*) dir->Get(key);
//  h2->Draw("colz");
//}


void Comparator::Draw( const char* key, Mode mode) {

  TH1::AddDirectory( false );
  TH1* h0 = Histo( key, 0);
  TH1* h1 = (TH1*) Histo( key, 1)->Clone("h1");

  TH1::AddDirectory( true );
  Draw( h0, h1, mode);    
}

  
void Comparator::Draw( const char* key0, const char* key1, Mode mode) {
  TH1* h0 = Histo( key0, 0);
  TH1* h1 = Histo( key1, 1);
    
  Draw( h0, h1, mode);
}

//   void Comparator::cd(const char* path ) {
//     path_ = path;
//   }
  

  
TH1* Comparator::Histo( const char* key, unsigned dirIndex) {
  if(dirIndex<0 || dirIndex>1) { 
    cerr<<"bad dir index: "<<dirIndex<<endl;
    return 0;
  }
  TDirectory* dir = 0;
  if(dirIndex == 0) dir = dir0_;
  if(dirIndex == 1) dir = dir1_;
  assert( dir );
  
  dir->cd();

  TH1* h = (TH1*) dir->Get(key);
  if(!h)  
    cerr<<"no key "<<key<<" in directory "<<dir->GetName()<<endl;
  return h;
}


void Comparator::Draw( TH1* h0, TH1* h1, Mode mode ) {
  if( !(h0 && h1) ) { 
    cerr<<"invalid histo"<<endl;
    return;
  }
    
  TH1::AddDirectory( false );
  h0_ = (TH1*) h0->Clone( "h0_");
  h1_ = (TH1*) h1->Clone( "h1_");
  TH1::AddDirectory( true );
    
  // unsetting the title, since the title of projections
  // is still the title of the 2d histo
  // and this is better anyway
  h0_->SetTitle("");
  h1_->SetTitle("");    

  //h0_->SetStats(1);
  //h1_->SetStats(1);

  if(rebin_>1) {
    h0_->Rebin( rebin_);
    h1_->Rebin( rebin_);
  }
  if(resetAxis_) {
    h0_->GetXaxis()->SetRangeUser( xMin_, xMax_);
    h1_->GetXaxis()->SetRangeUser( xMin_, xMax_);
  }

  TPaveStats *ptstats = new TPaveStats(0.7385057,0.720339,
				       0.9396552,0.8792373,"brNDC");
  ptstats->SetName("stats");
  ptstats->SetBorderSize(1);
  ptstats->SetLineColor(2);
  ptstats->SetFillColor(10);
  ptstats->SetTextAlign(12);
  ptstats->SetTextColor(2);
  ptstats->SetOptStat(1111);
  ptstats->SetOptFit(0);
  ptstats->Draw();
  h0_->GetListOfFunctions()->Add(ptstats);
  ptstats->SetParent(h0_->GetListOfFunctions());

  //std::cout << "FL: h0_->GetMean() = " << h0_->GetMean() << std::endl;
  //std::cout << "FL: h0_->GetRMS() = " << h0_->GetRMS() << std::endl;
  //std::cout << "FL: h1_->GetMean() = " << h1_->GetMean() << std::endl;
  //std::cout << "FL: h1_->GetRMS() = " << h1_->GetRMS() << std::endl;
  //std::cout << "FL: test2" << std::endl;
  TPaveStats *ptstats2 = new TPaveStats(0.7399425,0.529661,
					0.941092,0.6885593,"brNDC");
  ptstats2->SetName("stats");
  ptstats2->SetBorderSize(1);
  ptstats2->SetLineColor(4);
  ptstats2->SetFillColor(10);
  ptstats2->SetTextAlign(12);
  ptstats2->SetTextColor(4);
  TText *text = ptstats2->AddText("h1_");
  text->SetTextSize(0.03654661);

  std::ostringstream oss3;
  oss3 << h1_->GetEntries();
  const std::string txt_entries="Entries = "+oss3.str();
  text = ptstats2->AddText(txt_entries.c_str());
  std::ostringstream oss;
  oss << h1_->GetMean();
  const std::string txt_mean="Mean  = "+oss.str();
  text = ptstats2->AddText(txt_mean.c_str());
  std::ostringstream oss2;
  oss2 << h1_->GetRMS();
  const std::string txt_rms="RMS  = "+oss2.str();
  text = ptstats2->AddText(txt_rms.c_str());
  ptstats2->SetOptStat(1111);
  ptstats2->SetOptFit(0);
  ptstats2->Draw();
  h1_->GetListOfFunctions()->Add(ptstats2);
  ptstats2->SetParent(h1_->GetListOfFunctions());

  switch(mode) {
  case SCALE:
    h1_->Scale( h0_->GetEntries()/h1_->GetEntries() );
  case NORMAL:
    if(s0_)
      Styles::FormatHisto( h0_ , s0_);
    if(s1_)
      Styles::FormatHisto( h1_ , s1_);
      
    if( h1_->GetMaximum()>h0_->GetMaximum()) {
      h0_->SetMaximum( h1_->GetMaximum()*1.15 );
    }
    h0_->Draw();
    h1_->Draw("same");

    break;
  case EFF:
    h1_->Divide( h0_ );
    if(s1_)
      Styles::FormatHisto( h1_ , s0_);
    h1_->Draw();
  default:
    break;
  }
}


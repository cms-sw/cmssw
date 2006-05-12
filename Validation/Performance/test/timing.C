
// void test()
{

   gROOT->Reset() ;

#include "Riostream.h"

   ifstream in ;
   
   TString  Logo ;
   Int_t    Run, Evt ;
   TString  ModLabel ;
   TString  ModName ;
   Float_t  Time ;
      
  // in.clear() ;
   
   in.open("timing.log") ;
   
   TH1F* Vtx = new TH1F( "VtxSmeared", "VertexGenerator", 100, 0., 1. ) ;
   TH1F* OPr = new TH1F( "SimG4Object", "OscarProducer", 100, 0., 500. ) ;
   TH1F* Mix = new TH1F( "mix", "MixingModule",  100, 0., 1. ) ;
   TH1F* Pix = new TH1F( "pixdigi", "SiPixelDigitizer", 100, 0., 1. ) ;
   TH1F* SiS = new TH1F( "sistripdigi", "SiStripDigitizer", 100, 0., 1. ) ;
   TH1F* Eca = new TH1F( "ecaldigi", "EcalDigiProducer", 100, 0., 5. ) ;
   TH1F* Hca = new TH1F( "hcaldigi", "HcalDigiProcucer", 100, 0., 1. ) ;
   TH1F* CSC = new TH1F( "muoncscdigi", "CSCDigiProducer", 100, 0., 1. ) ;
   TH1F* DT  = new TH1F( "muondtdigi",  "DTDigitizer", 100, 0., 0.1 ) ;
   //
   // no PoolOutputModule in this chain 
   //
   
   while (1)
   {
      in >> Logo >> Evt >> Run >> ModLabel >> ModName >> Time ;
      if ( !in.good() ) break ;
      if ( Evt == 1 ) continue ; // skip init CPU for now...
      if ( ModName == "VertexGenerator" ) Vtx->Fill(Time) ;
      if ( ModName == "OscarProducer" ) OPr->Fill(Time) ;
      if ( ModName == "MixingModule" )  Mix->Fill(Time) ;
      if ( ModName == "SiPixelDigitizer" ) Pix->Fill(Time) ;
      if ( ModName == "SiStripDigitizer" ) SiS->Fill(Time) ;
      if ( ModName == "EcalDigiProducer" ) Eca->Fill(Time) ;
      if ( ModName == "HcalDigiProducer" ) Hca->Fill(Time) ;
      if ( ModName == "CSCDigiProducer" ) CSC->Fill(Time) ;
      if ( ModName == "DTDigitizer" ) DT->Fill(Time) ;
   }
   
   in.close() ;
   in.clear() ;
   
   TCanvas* c2 = new TCanvas("c2") ;
   OPr->Draw() ;
   OPr->Print("OscarProducer.ps") ;
   c2->SaveAs("OscarProducer.ps") ;
   
   TCanvas* c3 = new TCanvas( "c3", " ", 900, 900 ) ;
   c3->Divide(2,4) ;
   c3->cd(1) ;
   Vtx->Draw() ;
   c3->cd(2) ;
   Pix->Draw() ;
   c3->cd(3) ;
   SiS->Draw() ;
   c3->cd(4) ;
   Eca->Draw() ;
   c3->cd(5) ;
   Hca->Draw() ;
   c3->cd(6) ;
   CSC->Draw() ;
   c3->cd(7) ;
   DT->Draw() ;
   c3->cd(8) ;
   Mix->Draw() ;

   c3->SaveAs("Modules.ps")  ;
   
}

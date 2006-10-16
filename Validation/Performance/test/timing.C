#include "TFile.h"
// #include "TTree.h"
// #include "TBranch.h"
#include "TH1.h"
#include "TString.h"

#include "Riostream.h"

void timing( TString outfile="timing_minbias.root" )
{

   gROOT->Reset() ;

   ifstream in ;
   
   TString Type, Unt, Sprt;
   Float_t OprFreq ;
   
   
   TString  Logo ;
   Int_t    Run, Evt ;
   TString  ModLabel ;
   TString  ModName ;
   Float_t  Time ;      
   
   
   TFile* OutFile = new TFile( outfile, "RECREATE") ;
   
   TH1F* CPUInfo = new TH1F( "CPUInfo", "CPUInfo", 100, 0., 10. ) ;
    
   TH1F* Vtx = new TH1F( "VtxSmeared", "VertexGenerator", 100, 0., 0.1 ) ;
   TH1F* OPr = new TH1F( "SimG4Object", "OscarProducer", 100, 0., 500. ) ;
   TH1F* Mix = new TH1F( "mix", "MixingModule",  100, 0., 0.5 ) ;
   TH1F* Pix = new TH1F( "pixdigi", "SiPixelDigitizer", 100, 0., 0.5 ) ;
   TH1F* SiS = new TH1F( "sistripdigi", "SiStripDigitizer", 100, 0., 2. ) ;
   TH1F* Eca = new TH1F( "ecaldigi", "EcalDigiProducer", 100, 0., 2. ) ;
   TH1F* Hca = new TH1F( "hcaldigi", "HcalDigiProcucer", 100, 0., 0.5 ) ;
   TH1F* CSC = new TH1F( "muoncscdigi", "CSCDigiProducer", 100, 0., 0.1 ) ;
   TH1F* DT  = new TH1F( "muondtdigi",  "DTDigitizer",     100, 0., 0.1 ) ;
   TH1F* RPC = new TH1F( "muonrpcdigi", "RPCDigiProducer", 100, 0., 0.1 ) ;
   //
   // no PoolOutputModule in this chain 
   //
   
   in.open("cpu_info.log") ;
   // one is enough...
   in >> Type >> Unt >> Sprt >> OprFreq ;
   in.close() ;
   in.clear() ;
   
   CPUInfo->Fill( OprFreq/1000. ) ;
   
   in.open("timing_minbias.log") ;

   while (1)
   {
      in >> Logo >> Evt >> Run >> ModLabel >> ModName >> Time ;
      if ( !in.good() ) break ;
      if ( Evt == 1 ) continue ; // skip init CPU for now...
      if ( ModName == "VertexGenerator" )  Vtx->Fill(Time) ;
      if ( ModName == "OscarProducer" )    OPr->Fill(Time) ;
      if ( ModName == "MixingModule" )     Mix->Fill(Time) ;
      if ( ModName == "SiPixelDigitizer" ) Pix->Fill(Time) ;
      if ( ModName == "SiStripDigitizer" ) SiS->Fill(Time) ;
      if ( ModName == "EcalDigiProducer" ) Eca->Fill(Time) ;
      if ( ModName == "HcalDigiProducer" ) Hca->Fill(Time) ;
      if ( ModName == "CSCDigiProducer" )  CSC->Fill(Time) ;
      if ( ModName == "DTDigitizer" )      DT->Fill(Time) ;
      if ( ModName == "RPCDigiProducer" )  RPC->Fill(Time) ;
   }
   
   in.close() ;
   in.clear() ;
   
   cout << "[OVAL] : proc/cpuinfo = " << CPUInfo->GetMean()*1000. << Unt << endl ;

   cout << "[OVAL] : OscarProducer - MEAN CPU = " << OPr->GetMean() << " sec/evt" << endl ;

   cout << "[OVAL] : VertexGenerator - MEAN CPU = " << Vtx->GetMean() << " sec/evt" << endl ;
   cout << "[OVAL] : SiPixelDigitizer - MEAN CPU = " << Pix->GetMean() << " sec/evt" << endl ;
   cout << "[OVAL] : SiStripDigitizer - MEAN CPU = " << SiS->GetMean() << " sec/evt" << endl ;
   cout << "[OVAL] : EcalDigiProducer - MEAN CPU = " << Eca->GetMean() << " sec/evt" << endl ;
   cout << "[OVAL] : HcalDigiProducer - MEAN CPU = " << Hca->GetMean() << " sec/evt" << endl ;
   cout << "[OVAL] : CSCDigiProducer - MEAN CPU = " << CSC->GetMean() << " sec/evt" << endl ;
   cout << "[OVAL] : DTDigitizer - MEAN CPU = " << DT->GetMean() << " sec/evt" << endl ;
   cout << "[OVAL] : RPCDigiProducer - MEAN CPU = " << RPC->GetMean() << " sec/evt" << endl ;
   cout << "[OVAL] : MixingModule - MEAN CPU = " << Mix->GetMean() << " sec/evt" << endl ;
   
/*
   TCanvas* c2 = new TCanvas("c2") ;
   OPr->Draw() ;
   OPr->Print("OscarProducer.ps") ;
   c2->SaveAs("OscarProducer_minbias.ps") ;
   
   TCanvas* c3 = new TCanvas( "c3", " ", 900, 900 ) ;
   c3->Divide(2,4) ;
   c3->cd(1) ;
   Pix->Draw() ;
   c3->cd(2) ;
   SiS->Draw() ;
   c3->cd(3) ;
   Eca->Draw() ;
   c3->cd(4) ;
   Hca->Draw() ;
   c3->cd(5) ;
   CSC->Draw() ;
   c3->cd(6) ;
   DT->Draw() ;
   c3->cd(7) ;
   RPC->Draw() ;
   c3->cd(8) ;
   Mix->Draw() ;

   c3->SaveAs("Modules_minbias.ps")  ;
*/
   
   OutFile->Write() ;
         
}



/*
 * Produce plots for documentation of DT performance on the web.
 * The root files are opened and gif pictures are produced and placed
 * in the directory specified by the user.
 * The results are directly browserable at the address:
 * http://cmsdoc.cern.ch/cms/Physics/muon/CMSSW/Performance/DT/DTLocalRecoQualityTest/
 *
 * Intructions to run the macro:
 *    root
 *    .x produceWWWDoc.r
 *
 * NOTE: The user must have right access on afs dir:
 * /afs/cern.ch/cms/Physics/muon/CMSSW/Performance/DT/DTLocalRecoQualityTest
 *
 *
 * Author G. Cerminara - INFN Torino
 *
 */


{
  // Ask for the directory name (only the last part, es: CMSSW_1_2_0)
  TString nameDir;
  cout << "Set the name of the www directory: " << endl;
  cin >> nameDir;

  // Load 1D RecHits histos
  TFile *file1 = new TFile("DTRecHitQualityPlots.root");
  gROOT->LoadMacro("plotHitReso.r"); 
  plotWWWHitReso(1, nameDir);
  // Close all open canvases  
  TIter iter(gROOT->GetListOfCanvases());
  TCanvas *c;
  while( (c = (TCanvas *)iter()) ) {
      cout << "Closing " << c->GetName() << endl;
      c->Close();
    }
  file1->Close();

  // Load 2D Segments histos
  TFile *file2 = new TFile("DTSeg2DQualityPlots.root");
  plotWWWHitReso(2, nameDir);
  // Close all open canvases  
  TIter iter(gROOT->GetListOfCanvases());
  TCanvas *c;
  while( (c = (TCanvas *)iter()) ) {
      cout << "Closing " << c->GetName() << endl;
      c->Close();
    }
  file2->Close();

 // Load 2D SuperPhi Segments histos
  TFile *file3 = new TFile("DTSeg2DSLPhiQualityPlots.root");
  plotWWWHitReso(3, nameDir);
  // Close all open canvases  
  TIter iter(gROOT->GetListOfCanvases());
  TCanvas *c;
  while( (c = (TCanvas *)iter()) ) {
      cout << "Closing " << c->GetName() << endl;
      c->Close();
    }
  file3->Close();


  // Load 4D Segments histos
  TFile *file4 = new TFile("DTSeg4DQualityPlots.root");
  plotWWWHitReso(4, nameDir);
  // Close all open canvases  
  TIter iter(gROOT->GetListOfCanvases());
  TCanvas *c;
  while( (c = (TCanvas *)iter()) ) {
      cout << "Closing " << c->GetName() << endl;
      c->Close();
    }
  file4->Close();

}

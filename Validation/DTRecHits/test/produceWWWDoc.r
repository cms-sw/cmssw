

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
  TString dirBase = "/afs/cern.ch/cms/Physics/muon/CMSSW/Performance/DT/DTLocalRecoQualityTest/";
  // Ask for the directory name (only the last part, es: CMSSW_1_2_0)
  TString nameDir;
  cout << "Set the name of the www directory: " << endl;
  cin >> nameDir;

  // Load 1D RecHits histos
  TFile *file = new TFile("Cosmics_V0001_CMSSW_3_2_0.root");
  gROOT->LoadMacro("plotHitReso.r"); 
  gROOT->LoadMacro("plotHitPull.r"); 
  
  if(file->IsOpen()) {
    // plot residuals
    plotWWWHitReso(dirBase, 1, nameDir);
    // Close all open canvases  
    TIter iter(gROOT->GetListOfCanvases());
    TCanvas *c;
    while( (c = (TCanvas *)iter()) ) {
      cout << "Closing " << c->GetName() << endl;
      c->Close();
    }

    // plot pulls
    plotWWWHitPull(dirBase, 1, nameDir);
    // Close all open canvases  
    TIter iter(gROOT->GetListOfCanvases());
    TCanvas *c;
    while( (c = (TCanvas *)iter()) ) {
      cout << "Closing " << c->GetName() << endl;
      c->Close();
    }
    //    file1->Close();
  }
  

/*
  // Load 2D Segments histos
  //TFile *file2 = new TFile("DTSeg2DQualityPlots.root");
  if(file2->IsOpen()) {
    plotWWWHitReso(dirBase, 2, nameDir);
    // Close all open canvases  
    TIter iter(gROOT->GetListOfCanvases());
    TCanvas *c;
    while( (c = (TCanvas *)iter()) ) {
      cout << "Closing " << c->GetName() << endl;
      c->Close();
    }


    // plot pulls
    plotWWWHitPull(dirBase, 2, nameDir);
    // Close all open canvases  
    TIter iter(gROOT->GetListOfCanvases());
    TCanvas *c;
    while( (c = (TCanvas *)iter()) ) {
      cout << "Closing " << c->GetName() << endl;
      c->Close();
    }


    file2->Close();
  }
*/

 // Load 2D SuperPhi Segments histos
  //TFile *file3 = new TFile("DTSeg2DSLPhiQualityPlots.root");
  if(file->IsOpen()) {
    plotWWWHitReso(dirBase, 3, nameDir);
    // Close all open canvases  
    TIter iter(gROOT->GetListOfCanvases());
    TCanvas *c;
    while( (c = (TCanvas *)iter()) ) {
      cout << "Closing " << c->GetName() << endl;
      c->Close();
    }


    // plot pulls
    plotWWWHitPull(dirBase, 3, nameDir);
    // Close all open canvases  
    TIter iter(gROOT->GetListOfCanvases());
    TCanvas *c;
    while( (c = (TCanvas *)iter()) ) {
      cout << "Closing " << c->GetName() << endl;
      c->Close();
    }


    //file3->Close();
  }


  // Load 4D Segments histos
  //TFile *file4 = new TFile("DTSeg4DQualityPlots.root");
  if(file->IsOpen()) {
    plotWWWHitReso(dirBase, 4, nameDir);
    // Close all open canvases  
    TIter iter(gROOT->GetListOfCanvases());
    TCanvas *c;
    while( (c = (TCanvas *)iter()) ) {
      cout << "Closing " << c->GetName() << endl;
      c->Close();
    }

    // plot pulls
    plotWWWHitPull(dirBase, 4, nameDir);
    // Close all open canvases  
    TIter iter(gROOT->GetListOfCanvases());
    TCanvas *c;
    while( (c = (TCanvas *)iter()) ) {
      cout << "Closing " << c->GetName() << endl;
      c->Close();
    }



    file->Close();
  }
}

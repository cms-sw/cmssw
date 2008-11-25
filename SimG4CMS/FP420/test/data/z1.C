{

   //======================================================================
      printf("z1: gROOT Reset \n");
        gROOT->Reset();
        gROOT->SetStyle("Plain");
        //gStyle->SetOptStat(0);   //  no statistics _or_
        gStyle->SetOptStat(11111111);
        //
          gStyle->SetStatX(0.98);
          gStyle->SetStatY(0.99);
          gStyle->SetStatW(0.30);
          gStyle->SetStatH(0.25);
        //

        Float_t LeftOffset = 0.12;
        Float_t TopOffset = 0.12;

        gStyle->SetLineWidth(1);
        gStyle->SetErrorX(0);

//---=[ Titles,Labels ]=-----------
        gStyle->SetOptTitle(0);             // title on/off
        //      gStyle->SetTitleColor(0);           // title color
        gStyle->SetTitleColor(1);           // title color
        //      gStyle->SetTitleX(0.35);            // title x-position
        gStyle->SetTitleX(0.15);            // title x-position
        gStyle->SetTitleH(0.15);             // title height
        //      gStyle->SetTitleW(0.53);            // title width
        gStyle->SetTitleW(0.60);            // title width
        gStyle->SetTitleFont(42);           // title font
        gStyle->SetTitleFontSize(0.07);     // title font size

//---=[ Histogram style ]=----------
//      gStyle->SetHistFillColor(38);
                gStyle->SetFrameFillColor(41);

//---=[ Pad style ]=----------------
        gStyle->SetPadTopMargin(TopOffset);
        gStyle->SetPadBottomMargin(LeftOffset);
        gStyle->SetPadRightMargin(TopOffset);
        gStyle->SetPadLeftMargin(LeftOffset);
    //======================================================================
//
// Connect the input file and get the 2-d histogram in memory
    //======================================================================



//  TBrowser *b = new TBrowser
// macro to recreate a H1Tree
// using the H1EventList in skel_writelist.C
//
//      TFile *hfile = new TFile("newntfp420.root", "READ");     //open file
      TFile *hfile = new TFile("TheAnlysis.root", "READ");     //open file
      printf("z1: root file TheAnlysis is Open    \n");
    hfile.ls();
    hfile->Print();


//    getchar();

//    TPostScript* psfile = new TPostScript("zhplot.ps",111);
    TPostScript psfile ("z1.ps", 111);
    //======================================================================
//    c1 = new TCanvas("c1"," ",200,10,600,480);
//    c1->Clear();
//    TPad pad1("pad1"," ",0.1,0.6,0.6,0.9);
//    TPad pad2("pad2"," ",0.6,0.9,1.6,1.9);
//    TPad pad3("pad3"," ",0.1,0.6,0.1,0.4);
//    TPad pad4("pad4"," ",0.6,0.9,0.1,0.4);

//    pad1.Draw();
//    pad1.cd;
//    hfile.Draw("LAr_H_Q2s");

//    pad2.Draw();
//    pad2.cd;
//    hfile.Draw("LAr_H_Epz");

//    pad3.Draw();
//    pad3.cd;
//    hfile.Draw("LAr_H_Ys");

//    pad4.Draw();
//    pad4.cd;
//    hfile.Draw("LAr_elecE");
    //======================================================================
//    c1 = new TCanvas("c1"," ");
//     hfile.cd();
//        c1.Divide(1,2); //automatic pad generation
//      c1_1.cd();
//      c1_1.Draw();
//      hfile.Draw("SumEDep");
//      c1_2.cd();
//      c1_2.Draw();
//      hfile.Draw("TrackL");
//      c1_3.cd();
//      c1_3.Draw();
//      hfile.Draw("LAr_H_Ys");
//      c1_4.cd();
//      c1_4.Draw();
//      hfile.Draw("LAr_elecE");
//    c1->Update();
    //======================================================================
    TCanvas* c1 = new TCanvas("c1", "FP420Analysis", 600, 800);
     hfile.cd();
    //======================================================================111111
     c1->Update();
    c1->Clear();
    c1->Divide(1,3);
    c1->cd(1); 
    SumEDep->Draw();

    c1->cd(2); 
    TrackL->Draw();

    c1->cd(3); 
    NumberOfHits->Draw();

    c1->Update();

    //======================================================================222222
     c1->Clear();
     c1->Divide(3,2); 
     c1->cd(1); 
      TH1F *mpelec = (TH1F*)hfile->Get("VtxX");
		mpelec->SetLineColor(3);
		mpelec->SetMarkerStyle(20);
		mpelec->SetMarkerSize(0.4);
		mpelec->GetYaxis()->SetLabelSize(0.04);
		mpelec->SetMarkerColor(kBlue);
		mpelec->Draw("Error");

     c1->cd(2); 
      TH1F *mpelec1= (TH1F*)hfile->Get("VtxY");
       mpelec1->SetMarkerStyle(20);
       mpelec1->SetMarkerSize(0.4);
       mpelec1->GetYaxis()->SetLabelSize(0.04);
       mpelec1->SetMarkerColor(kBlue);
       mpelec1->SetLineColor(3);
       mpelec1->Draw("Error");

     c1->cd(3); 
      TH1F *mpelec2= (TH1F*)hfile->Get("VtxZ");
       mpelec2->SetMarkerStyle(20);
       mpelec2->SetMarkerSize(0.4);
       mpelec2->GetYaxis()->SetLabelSize(0.04);
       mpelec2->SetMarkerColor(kBlue);
       mpelec2->SetLineColor(3);
       mpelec2->Draw("Error");

     c1->cd(4); 
      TH1F *mpelec3= (TH1F*)hfile->Get("PrimaryEta");
       mpelec3->SetMarkerStyle(20);
       mpelec3->SetMarkerSize(0.4);
       mpelec3->GetYaxis()->SetLabelSize(0.04);
       mpelec3->SetMarkerColor(kBlue);
       mpelec3->SetLineColor(3);
       mpelec3->Draw("Error");

     c1->cd(5); 
      TH1F *mpelec3= (TH1F*)hfile->Get("PrimaryPhigrad");
       mpelec3->SetMarkerStyle(20);
       mpelec3->SetMarkerSize(0.4);
       mpelec3->GetYaxis()->SetLabelSize(0.04);
       mpelec3->SetMarkerColor(kBlue);
       mpelec3->SetLineColor(3);
       mpelec3->Draw("Error");

     c1->cd(6); 
      TH1F *mpelec3= (TH1F*)hfile->Get("PrimaryTh");
       mpelec3->SetMarkerStyle(20);
       mpelec3->SetMarkerSize(0.4);
       mpelec3->GetYaxis()->SetLabelSize(0.04);
       mpelec3->SetMarkerColor(kBlue);
       mpelec3->SetLineColor(3);
       mpelec3->Draw("Error");


     c1->Update();




    //======================================================================333333
     c1->Clear();
     c1->Divide(1,3); 
     c1->cd(1); 
      TH1F *mpelec = (TH1F*)hfile->Get("PrimaryLastpoX");
		mpelec->SetLineColor(3);
		mpelec->SetMarkerStyle(20);
		mpelec->SetMarkerSize(0.4);
		mpelec->GetYaxis()->SetLabelSize(0.04);
		mpelec->SetMarkerColor(kBlue);
		mpelec->Draw("Error");


     c1->cd(2); 
      TH1F *mpelec1= (TH1F*)hfile->Get("PrimaryLastpoY");
       mpelec1->SetMarkerStyle(20);
       mpelec1->SetMarkerSize(0.4);
       mpelec1->GetYaxis()->SetLabelSize(0.04);
       mpelec1->SetMarkerColor(kBlue);
       mpelec1->SetLineColor(3);
       mpelec1->Draw("Error");

     c1->cd(3); 
      TH1F *mpelec2= (TH1F*)hfile->Get("PrimaryLastpoZ");
       mpelec2->SetMarkerStyle(20);
       mpelec2->SetMarkerSize(0.4);
       mpelec2->GetYaxis()->SetLabelSize(0.04);
       mpelec2->SetMarkerColor(kBlue);
       mpelec2->SetLineColor(3);
       mpelec2->Draw("Error");


     c1->Update();
    //======================================================================
    //======================================================================444444
     c1->Clear();
     c1->Divide(1,3); 
     c1->cd(1); 
      TH1F *mpelec2= (TH1F*)hfile->Get("zHits");
       mpelec2->SetMarkerStyle(20);
       mpelec2->SetMarkerSize(0.4);
       mpelec2->GetYaxis()->SetLabelSize(0.04);
       mpelec2->SetMarkerColor(kBlue);
       mpelec2->SetLineColor(3);
       mpelec2->Draw("Error");

     c1->cd(2); 
      TH1F *mpelec2= (TH1F*)hfile->Get("zHitsnoMI");
       mpelec2->SetMarkerStyle(20);
       mpelec2->SetMarkerSize(0.4);
       mpelec2->GetYaxis()->SetLabelSize(0.04);
       mpelec2->SetMarkerColor(kBlue);
       mpelec2->SetLineColor(3);
       mpelec2->Draw("Error");


     c1->cd(3); 
      TH1F *mpelec2= (TH1F*)hfile->Get("zHitsTrLoLe");
       mpelec2->SetMarkerStyle(20);
       mpelec2->SetMarkerSize(0.4);
       mpelec2->GetYaxis()->SetLabelSize(0.04);
       mpelec2->SetMarkerColor(kBlue);
       mpelec2->SetLineColor(3);
       mpelec2->Draw("Error");


     c1->Update();
    //======================================================================




    //======================================================================
    //======================================================================
    //======================================================================
    //======================================================================
    //======================================================================


                               //// wait
     //    getchar();

    //======================================================================


    psfile->Close();
    hfile->Close();

        //  Exit Root
        gSystem->Exit(0);

}

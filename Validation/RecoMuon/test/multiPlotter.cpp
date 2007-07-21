/**
 *
 *  Class: multiPlotter
 *
 *  Description:  
 *  This macro will draw histograms from a list of root files and write them
 *  to a target root file. The target file is newly created and must not be
 *  identical to one of the source files.
 *
 *  This code is based on the hadd.C example by Rene Brun and Dirk Geppert,
 *  which had a problem with directories more than one level deep.
 *  (see macro hadd_old.C for this previous implementation).
 *
 *  $Date: $
 *  $Revision: $
 *
 *  Authors:
 *  A. Everett Purdue University
 *  
 **/


#include <string.h>
#include <TROOT.h>
#include <TSystem.h>
#include <TStyle.h>
#include "TChain.h"
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TTree.h"
#include "TKey.h"
#include "Riostream.h"
#include "TCanvas.h"
#include <TPDF.h>
#include <TLegend.h>

//#include <boost/program_options.hpp>
#include <iostream>
#include <fstream>

TList *FileList;
TFile *Target;
TPDF *pdf;
TLegend *gLegend;

void drawLoop( TDirectory *target, TList *sourcelist, TCanvas *c1 );

int main(int argc, char *argv[] )
{

  TDirectory *target = TFile::Open(TString(argv[1])+".root","RECREATE");

  TList *sourcelist = new TList();  
  for (int i = 2; i < argc; i++) {
    cout << argv[i] << " " << endl;
    sourcelist->Add(TFile::Open(argv[i]));
  }
  
  //sourcelist->Print();

  bool writePdf = true;
  TCanvas* c1 = new TCanvas("c1") ;
  pdf = 0 ;
  if (writePdf) pdf = new TPDF(TString(argv[1])+".pdf") ;
  int pageNumber = 2 ;
  double titleSize = 0.050 ; 
  
  gROOT->SetStyle("Plain") ; 
  gStyle->SetPalette(1) ; 
  //gStyle->SetOptStat(111111) ;
  gStyle->SetOptStat(0) ;
  c1->UseCurrentStyle() ; 
  gROOT->ForceStyle() ;
  
  drawLoop(target,sourcelist,c1);
  
  if (writePdf) pdf->Close();
  target->Close();
  return 0;
}

void drawLoop( TDirectory *target, TList *sourcelist, TCanvas *c1 )
{

  TString path( (char*)strstr( target->GetPath(), ":" ) );
  path.Remove( 0, 2 );

  TFile *first_source = (TFile*)sourcelist->First();
  first_source->cd( path );
  TDirectory *current_sourcedir = gDirectory;
  //gain time, do not add the objects in the list in memory
  Bool_t status = TH1::AddDirectoryStatus();
  TH1::AddDirectory(kFALSE);

  // loop over all keys in this directory
  TChain *globChain = 0;
  TIter nextkey( current_sourcedir->GetListOfKeys() );
  TKey *key, *oldkey=0;
  while ( (key = (TKey*)nextkey())) {

    //keep only the highest cycle number for each key
    if (oldkey && !strcmp(oldkey->GetName(),key->GetName())) continue;

    // read object from first source file
    first_source->cd( path );
    TObject *obj = key->ReadObj();

    if ( obj->IsA()->InheritsFrom( "TH1" ) 
	 && !obj->IsA()->InheritsFrom("TH2") ) {

      // descendant of TH1 -> merge it
      gLegend = new TLegend(.85,.15,1.0,.30,"");
      gLegend->SetHeader(gDirectory->GetName());
      Color_t color = 1;
      Style_t style = 22;
      TH1 *h1 = (TH1*)obj;
      h1->SetLineColor(color);
      h1->SetMarkerStyle(style);
      h1->SetMarkerColor(color);
      h1->Draw();
      TString tmpName(first_source->GetName());
      gLegend->AddEntry(h1,tmpName.Remove(tmpName.Length()-5,5),"LP");
      c1->Update();

      // loop over all source files and add the content of the
      // correspondant histogram to the one pointed to by "h1"
      TFile *nextsource = (TFile*)sourcelist->After( first_source );
      while ( nextsource ) {
        
        // make sure we are at the correct directory level by cd'ing to path
        nextsource->cd( path );
        TKey *key2 = (TKey*)gDirectory->GetListOfKeys()->FindObject(h1->GetName());
        if (key2) {
           TH1 *h2 = (TH1*)key2->ReadObj();
	   color++;
	   style++;
	   h2->SetLineColor(color);
	   h2->SetMarkerStyle(style);
	   h2->SetMarkerColor(color);
           h2->Draw("same");
	   TString tmpName(nextsource->GetName());
	   gLegend->AddEntry(h2,tmpName.Remove(tmpName.Length()-5,5),"LP");
	   gLegend->Draw("same");
	   c1->Update();
           //- delete h2;
        }

        nextsource = (TFile*)sourcelist->After( nextsource );
      }
    }
    else if ( obj->IsA()->InheritsFrom( "TH2" ) ) {
      // descendant of TH2 -> merge it
      gLegend = new TLegend(.85,.15,1.0,.30,"");
      gLegend->SetHeader(gDirectory->GetName());
      Color_t color = 1;
      Style_t style = 22;
      TH2 *h1 = (TH2*)obj;
      h1->SetLineColor(color);
      h1->SetMarkerStyle(style);
      h1->SetMarkerColor(color);
      h1->Draw();
      TString tmpName(first_source->GetName());
      gLegend->AddEntry(h1,tmpName.Remove(tmpName.Length()-5,5),"LP");
      c1->Update();

      // loop over all source files and add the content of the
      // correspondant histogram to the one pointed to by "h1"
      TFile *nextsource = (TFile*)sourcelist->After( first_source );
      while ( nextsource ) {
        
        // make sure we are at the correct directory level by cd'ing to path
        nextsource->cd( path );
        TKey *key2 = (TKey*)gDirectory->GetListOfKeys()->FindObject(h1->GetName());
        if (key2) {
           TH2 *h2 = (TH2*)key2->ReadObj();
	   color++;
	   style++;
	   h2->SetLineColor(color);
	   h2->SetMarkerStyle(style);
	   h2->SetMarkerColor(color);
	   h2->Draw("same");
	   TString tmpName(nextsource->GetName());
	   gLegend->AddEntry(h2,tmpName.Remove(tmpName.Length()-5,5),"LP");
	   gLegend->Draw("same");
	   c1->Update();
           //- delete h2;
        }
        nextsource = (TFile*)sourcelist->After( nextsource );
      }
      //      c1->Update();c1->Write(obj->GetName(),TObject::kOverwrite);
    }
    else if ( obj->IsA()->InheritsFrom( "TTree" ) ) {
      cout << "I don't draw trees" << endl;
    } else if ( obj->IsA()->InheritsFrom( "TDirectory" ) ) {
      // it's a subdirectory
      cout << "Found subdirectory " << obj->GetName() << endl;
      // create a new subdir of same name and title in the target file
      target->cd();
      TDirectory *newdir = target->mkdir( obj->GetName(), obj->GetTitle() );

      // newdir is now the starting point of another round of merging
      // newdir still knows its depth within the target file via
      // GetPath(), so we can still figure out where we are in the recursion
      drawLoop( newdir, sourcelist, c1 );

    } else {

      // object is of no type that we know or can handle
      cout << "Unknown object type, name: " 
           << obj->GetName() << " title: " << obj->GetTitle() << endl;
    }
 
    // now write the merged TCanvas (which is "in" obj) to the target file
    // note that this will just store obj in the current directory level,
    // which is not persistent until the complete directory itself is stored
    // by "target->Write()" below
    if ( obj ) {
      target->cd();

      //!!if the object is a tree, it is stored in globChain...
	if(obj->IsA()->InheritsFrom( "TTree" ))
          globChain->Merge(target->GetFile(),0,"keep");
	else {
	  //- obj->Write( key->GetName() );
	}
	if ( obj->IsA()->InheritsFrom( "TH1" ) ) {
	  c1->Write( obj->GetName() );
	}
    }

  } // while ( ( TKey *key = (TKey*)nextkey() ) )

  // save modifications to target file
  target->SaveSelf(kTRUE);
  TH1::AddDirectory(status);
}

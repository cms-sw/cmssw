/**
 *
 *  Class: multiPlotter
 *
 *  Description:  
 *  This macro will draw histograms from a list of root files and write them
 *  to a target root file. The target file is newly created and must not be
 *  identical to one of the source files.
 *  Option: create a pdf file and / or a directory of gif files
 *
 *  This code is based on the hadd.C example by Rene Brun and Dirk Geppert,
 *  which had a problem with directories more than one level deep.
 *  (see macro hadd_old.C for this previous implementation).
 *
 *  $Date: 2010/11/29 20:42:02 $
 *  $Revision: 1.9 $
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
#include <TGraph.h>

#include <boost/program_options.hpp>
#include <iostream>
#include <fstream>

TList *FileList;
TFile *Target;
TPDF *pdf;
TLegend *gLegend;
TString *baseName;
bool makeGraphic;

void drawLoop( TDirectory *target, TList *sourcelist, TCanvas *c1 );

int main(int argc, char *argv[] )
{

  // Default arguments
  std::string outbase = "canvas";
  std::string outname = outbase + ".root";
  std::string pdfname = outbase + ".pdf";
  bool makePdf = false;
  makeGraphic = false;
  std::string infileName;
  std::vector<std::string> inFileVector ; inFileVector.clear() ; 

  //--- Get parameters from command line ---//
  boost::program_options::options_description desc(
	      "Available options for multiPlotter") ;
  desc.add_options()
    ("help,h","Print this help message")
    ("infile,i",   boost::program_options::value<std::string>(),
     //     "Input file name (Default is validation.root)") 
     "Input file name") 
    ("outfile,o",  boost::program_options::value<std::string>(),
     "Sets output files to <outfile>.root/.pdf (default is canvas)")
    ("pdf,p",
     "Make a PDF file")
    ("graphic,g",
     "makes a gif file for each TCanvas");

  std::string usage = "\nSample multiPlotter usage::\n" ; 
  usage += "\"multiPlotter -o validation-canvas -i validation.root\"\n " ; 
  usage += "\t input= validation.root\n" ;
  usage += "\t output= validation-canvas.root\n" ;
  usage += "\"multiPlotter -g -p -o validation-canvas -i \"validation_01.root validation_02.root\" \"\n" ;
  usage += "\t input= validation_01.root AND validation_02.root\n" ;
  usage += "\t output= validation-canvas.root\n" ;
  usage += "\t         validation-canvas.pdf\n" ;
  usage += "\t         gif files in validation-canvas/ \n\t\t\t (a directory tree which has the same \n\t\t\t directory structure as validation-canvas.root\n" ;
  usage += "\n" ; 

  boost::program_options::positional_options_description pos ; 
  boost::program_options::variables_map vmap ;
  
  try {
    boost::program_options::store(boost::program_options::command_line_parser(argc,argv).
				  options(desc).positional(pos).run(), vmap) ; 
  } catch (boost::program_options::error const& x) {
    std::cerr << "Unable to parse options:\n"
	      << x.what() << "\n\n" ;
    std::cerr << desc << usage << std::endl ;
    return 1 ; 
  }
  
  boost::program_options::notify(vmap) ; 
  if (vmap.count("help")) {
    std::cout << desc << usage <<  std::endl ;
    return 1 ;
  }
  if (vmap.count("outfile")) {
    outbase = vmap["outfile"].as<std::string>() ; 
    outname = outbase + ".root" ;
    pdfname = outbase + ".pdf" ;
  }
  if (vmap.count("pdf")) {
    makePdf = true ; 
  } 
  if (vmap.count("graphic")) {
    makeGraphic = true ; 
  } 
  if (vmap.count("infile")) {
    infileName = vmap["infile"].as<std::string>() ;
    /*
    ifstream inFile(infileName.c_str()) ;
    if (inFile.is_open()) { //--- input files listed in a file ---//
      while ( !inFile.eof() ) {
	std::string skipped ;
	getline(inFile,skipped) ; 
	inFileVector.push_back( skipped ) ;
      }
    } else 
    */
    { //--- Assume the file is a space-separated list of files -//
      size_t strStart = 0 ; 
      for (size_t itr=infileName.find(" ",0); itr!=std::string::npos;
	   itr=infileName.find(" ",itr)) {
	std::string skipped = infileName.substr(strStart,(itr-strStart)) ; 
	itr++ ; strStart = itr ; 
	inFileVector.push_back( skipped ) ;
      }
      //--- Fill the last entry ---//
      inFileVector.push_back( infileName.substr(strStart,infileName.length()) ); 
    }
  }
  else {
    cout << " *** No input file given: please define one " << endl;
    return 0;
  }

  TDirectory *target = TFile::Open(TString(outname),"RECREATE");
  
  baseName = new TString(outbase);
  baseName->Append("/");
  
  TList *sourcelist = new TList();  
  for (std::vector<std::string>::size_type i = 0; i < inFileVector.size(); i++) {
    cout << inFileVector[i] << " " << endl;
    sourcelist->Add(TFile::Open(TString(inFileVector[i])));
  }

  TCanvas* c1 = new TCanvas("c1") ;
  pdf = 0 ;
  if (makePdf) pdf = new TPDF(TString(pdfname)) ;
  //  int pageNumber = 2 ;
  // double titleSize = 0.050 ; 
  
  gROOT->SetStyle("Plain") ; 
  gStyle->SetPalette(1) ; 
  //gStyle->SetOptStat(111111) ;
  gStyle->SetOptStat(0) ;
  c1->UseCurrentStyle() ; 
  gROOT->ForceStyle() ;
  
  drawLoop(target,sourcelist,c1);
  
  if (makePdf) pdf->Close();
  target->Close();
  return 0;
}

void drawLoop( TDirectory *target, TList *sourcelist, TCanvas *c1 )
{

  TString path( (char*)strstr( target->GetPath(), ":" ) );
  path.Remove( 0, 2 );

  TString sysString(path);sysString.Prepend(baseName->Data());

  TFile *first_source = (TFile*)sourcelist->First();
  first_source->cd( path );
  TDirectory *current_sourcedir = gDirectory;
  //gain time, do not add the objects in the list in memory
  Bool_t status = TH1::AddDirectoryStatus();
  TH1::AddDirectory(kFALSE);

  // loop over all keys in this directory
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
      gLegend = new TLegend(.7,.15,.95,.4,"");
      gLegend->SetHeader(gDirectory->GetName());
      Color_t color = 1;
      Style_t style = 22;
      TH1 *h1 = (TH1*)obj;
      h1->SetLineColor(color);
      h1->SetMarkerStyle(style);
      h1->SetMarkerColor(color);
      h1->Draw();
      TString tmpName(first_source->GetName());
      gLegend->AddEntry(h1,tmpName,"LP");
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
	   gLegend->AddEntry(h2,tmpName,"LP");
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
      gLegend->AddEntry(h1,tmpName,"LP");
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
	   gLegend->AddEntry(h2,tmpName,"LP");
	   gLegend->Draw("same");
	   c1->Update();
           //- delete h2;
        }
        nextsource = (TFile*)sourcelist->After( nextsource );
      }
    }
    else if ( obj->IsA()->InheritsFrom( "TGraph" ) ) {
      obj->IsA()->Print();
      gLegend = new TLegend(.7,.15,.95,.4,"");
      gLegend->SetHeader(gDirectory->GetName());
      Color_t color = 1;
      Style_t style = 22;
      TGraph *h1 =(TGraph*)obj;
      h1->SetLineColor(color);
      h1->SetMarkerStyle(style);
      h1->SetMarkerColor(color);
      h1->GetHistogram()->Draw();
      h1->Draw();
      TString tmpName(first_source->GetName());
      gLegend->AddEntry(h1,tmpName,"LP");
      c1->Update();

      // loop over all source files and add the content of the
      // correspondant histogram to the one pointed to by "h1"
      TFile *nextsource = (TFile*)sourcelist->After( first_source );
      while ( nextsource ) {
        
        // make sure we are at the correct directory level by cd'ing to path
        nextsource->cd( path );
        TKey *key2 = (TKey*)gDirectory->GetListOfKeys()->FindObject(h1->GetName());
        if (key2) {
           TGraph *h2 = (TGraph*)key2->ReadObj();
	   color++;
	   style++;
	   h2->SetLineColor(color);
	   h2->SetMarkerStyle(style);
	   h2->SetMarkerColor(color);
           h2->Draw("same");
	   TString tmpName(nextsource->GetName());
	   gLegend->AddEntry(h2,tmpName,"LP");
	   gLegend->Draw("same");
	   c1->Update();
           //- delete h2;
        }

        nextsource = (TFile*)sourcelist->After( nextsource );
      }
    }
    else if ( obj->IsA()->InheritsFrom( "TTree" ) ) {
      cout << "I don't draw trees" << endl;
    } else if ( obj->IsA()->InheritsFrom( "TDirectory" ) ) {
      // it's a subdirectory
      cout << "Found subdirectory " << obj->GetName() << endl;

      // create a new subdir of same name and title in the target file
      target->cd();
      TDirectory *newdir = target->mkdir( obj->GetName(), obj->GetTitle() );

      // create a new subdir of same name in the file system
      TString newSysString(sysString+"/"+obj->GetName());
      if(makeGraphic) gSystem->mkdir(newSysString.Data(),kTRUE);

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

	if ( obj->IsA()->InheritsFrom( "TH1") || obj->IsA()->InheritsFrom("TGraph")) {
	  //	 && !obj->IsA()->InheritsFrom("TH2") ) {
	  TString newName(obj->GetName());
	  newName.ReplaceAll("(",1,"_",1);
	  newName.ReplaceAll(")",1,"_",1);
	  c1->SetName(newName);
	  c1->Write( c1->GetName(),TObject::kOverwrite );
	  
	  if(makeGraphic) {
	    if (gROOT->IsBatch()) {
	      c1->Print("temp.eps");
	      gSystem->Exec("pstopnm -ppm -xborder 0 -yborder 0 -portrait temp.eps");
	      char tempCommand[200];
	      sprintf(tempCommand,"ppmtogif temp.eps001.ppm > %s/%s.gif",sysString.Data(),c1->GetName());
	      gSystem->Exec(tempCommand);
	    } else {	    
	      c1->Print(sysString + "/" + TString(c1->GetName())+".gif");
	    }
	  }
	  
	}
    }
    //if(gLegend) delete gLegend;
  } // while ( ( TKey *key = (TKey*)nextkey() ) )

  // save modifications to target file
  target->SaveSelf(kTRUE);
  TH1::AddDirectory(status);
}

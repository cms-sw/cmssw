#include "Riostream.h"
#include <stdio.h>
#include <stdlib.h>


void mkTree() {

   std::string file("lumis.dat");

   cout << file << endl;

   ifstream in;
   in.open("lumis.dat"); 

   Float_t run,ls,lumiDelivered, lumiReported;
   Int_t nlines = 0;
   TFile *f = new TFile("lumis.root","RECREATE");
   TNtuple *ntuple = new TNtuple("ntuple","data from ascii file","run:ls:lumiDelivered:lumiReported");

   while (1) {
      in >> run >> ls >> lumiDelivered >> lumiReported;
      if (!in.good()) break;
      if (nlines < 5) printf("run=%8f, ls=%8f, lumiDelivered=%8f, lumiReported=%8f\n",run, ls, lumiDelivered, lumiReported);
      ntuple->Fill(run, ls, lumiDelivered, lumiReported);
      nlines++;
   }
   printf(" found %d points\n",nlines);

   in.close();

   f->Write();

   gROOT->ProcessLine(".q");
}


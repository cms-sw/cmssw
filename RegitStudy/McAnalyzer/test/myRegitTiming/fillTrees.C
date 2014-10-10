#include <iostream>

#include "TFile.h"
#include "TTree.h"

using namespace std;

void fillTrees(const char* timereport, const char* peakvsize, const char* timestats, const char* output)
{
   TFile *tf = new TFile(output, "RECREATE");

   TTree *trtr = new TTree("timereport","timereport");
   trtr->ReadFile(timereport,"evtcpu/F:evtreal:modruncpu:modrunreal:modvisitcpu:modvisitreal:modname/C");

   TTree *trpv = new TTree("peakvsize","peakvsize");
   trpv->ReadFile(peakvsize,"peakvsize/F");

   TTree *trts = new TTree("timestats","timestats");
   trts->ReadFile(timestats,"mint/F:maxt:avgt:tott:minc:maxc:avgc:totc");

   tf->Write();
   tf->Close();
}

void fillTrees(const char* timereport, const char* peakvsize, const char* timestats, const char* timemodule, const char* output)
{
   TFile *tf = new TFile(output, "RECREATE");

   TTree *trtr = new TTree("timereport","timereport");
   trtr->ReadFile(timereport,"evtcpu/F:evtreal:modruncpu:modrunreal:modvisitcpu:modvisitreal:modname/C");

   TTree *trpv = new TTree("peakvsize","peakvsize");
   trpv->ReadFile(peakvsize,"peakvsize/F");

   TTree *trts = new TTree("timestats","timestats");
   trts->ReadFile(timestats,"mint/F:maxt:avgt:tott:minc:maxc:avgc:totc");

   TTree *trtm = new TTree("timemodule","timemodule");
   trtm->ReadFile(timemodule,"modname/C:modtype/C:time/F");

   tf->Write();
   tf->Close();
}

void fillTrees(const char* timereport, const char* peakvsize, const char* timestats, const char* timemodule, const char* centrality, const char* output)
{
   TFile *tf = new TFile(output, "RECREATE");

   TTree *trtr = new TTree("timereport","timereport");
   trtr->ReadFile(timereport,"evtcpu/F:evtreal:modruncpu:modrunreal:modvisitcpu:modvisitreal:modname/C");

   TTree *trpv = new TTree("peakvsize","peakvsize");
   trpv->ReadFile(peakvsize,"peakvsize/F");

   TTree *trts = new TTree("timestats","timestats");
   trts->ReadFile(timestats,"mint/F:maxt:avgt:tott:minc:maxc:avgc:totc");

   TTree *trtm = new TTree("timemodule","timemodule");
   trtm->ReadFile(timemodule,"modname/C:modtype/C:time/F");

   TTree *trcent = new TTree("centrality","centrality");
   trcent->ReadFile(centrality,"centBin/I");

   tf->Write();
   tf->Close();
}

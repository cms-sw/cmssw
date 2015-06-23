{
   gROOT->SetStyle("Plain");
   cout << "loading..." <<endl;
   gSystem->Load("libCintex");
   Cintex::Enable();
   gSystem->Load("libFWCoreFWLite");
   FWLiteEnabler::enable();
   gSystem->Load("libRooFit.so");
   using namespace RooFit;
}

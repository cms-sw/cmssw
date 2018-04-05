{
   gROOT->SetStyle("Plain");
   cout << "loading..." <<endl;
   gSystem->Load("libFWCoreFWLite");
   FWLiteEnabler::enable();
   gSystem->Load("libRooFit.so");
   using namespace RooFit;
}

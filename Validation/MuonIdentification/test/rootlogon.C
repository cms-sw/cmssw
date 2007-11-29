{
   gROOT->SetStyle("Plain");
   cout << "loading..." <<endl;
   gSystem->Load("libCintex");
   Cintex::Enable();
   gSystem->Load("libFWCoreFWLite");
   AutoLibraryLoader::enable();
   gSystem->Load("libRooFit.so");
   using namespace RooFit;
}

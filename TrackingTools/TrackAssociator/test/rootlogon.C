{
   gROOT->SetStyle("Plain");
   cout << "loading..." <<endl;
   gSystem->Load("libFWCoreFWLite");
   AutoLibraryLoader::enable();
   gSystem->Load("libRooFit.so");
   using namespace RooFit;
}

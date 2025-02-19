{
gSystem->Load("libCintex");
Cintex::Enable();
cout << "Loading FWLite..." << endl;
gSystem->Load("libFWCoreFWLite");
AutoLibraryLoader::enable();

} 

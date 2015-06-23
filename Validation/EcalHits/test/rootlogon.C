{
gSystem->Load("libCintex");
Cintex::Enable();
cout << "Loading FWLite..." << endl;
gSystem->Load("libFWCoreFWLite");
FWLiteEnabler::enable();
gSystem->Load("libSimDataFormatsEcalValidation.so");
gSystem->Load("libSimDataFormatsTrack.so");
gSystem->Load("libSimDataFormatsVertex.so");
}

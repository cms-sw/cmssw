{
cout << "Loading FWLite..." << endl;    // load CMSSW libraries
gSystem->Load("libFWCoreFWLite");
AutoLibraryLoader::enable();

cout << "Setting Style to Plain..." << endl;
gROOT->SetStyle("Plain");        // Switches off the ROOT default style
//gPad->UseCurrentStyle();       // this makes everything black and white,
                                 // removing the default red border on images
//gROOT->ForceStyle();           // forces the style chosen above to be used,
                                 // not the style the rootfile was made with

cout << "Setting Palette to 1..." << endl;
gStyle->SetPalette(1);         // get better colors than default
}

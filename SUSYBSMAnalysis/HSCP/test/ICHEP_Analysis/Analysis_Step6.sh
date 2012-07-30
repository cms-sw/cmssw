root -l -b << EOF
   TString makeshared(gSystem->GetMakeSharedLib());
   TString dummy = makeshared.ReplaceAll("-W ", "-Wno-deprecated-declarations -Wno-deprecated ");
   TString dummy = makeshared.ReplaceAll("-Wshadow ", " -std=c++0x ");
   cout << "Compilling with the following arguments: " << makeshared << endl;
   gSystem->SetMakeSharedLib(makeshared);
   gSystem->SetIncludePath( "-I$ROOFITSYS/include" );
//  .x Analysis_Step6.C++("ANALYSE", "Results/dedxASmi/combined/Eta15/PtMin50/Type0/" ,"Gluino500_f10");
//  .x Analysis_Step6.C++("COMPILE", "Results/dedxASmi/combined/Eta15/PtMin50/Type0/" ,"Gluino500_f10");
  .x Analysis_Step6.C++("Final", "", "");
//  .x Analysis_Step6.C+ ("Final", "", "", -1, -1, -1, "_SystP");
//  .x Analysis_Step6.C+ ("Final", "", "", -1, -1, -1, "_SystI");
//  .x Analysis_Step6.C+ ("Final", "", "", -1, -1, -1, "_SystM");
//  .x Analysis_Step6.C+ ("Final", "", "", -1, -1, -1, "_SystT");
EOF


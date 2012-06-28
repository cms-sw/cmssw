root -l -b << EOF
   TString makeshared(gSystem->GetMakeSharedLib());
   TString dummy = makeshared.ReplaceAll("-W ", "");
   TString dummy = makeshared.ReplaceAll("-Wshadow ", " -std=c++0x ");
   gSystem->SetMakeSharedLib(makeshared);
   gSystem->SetIncludePath( "-I$ROOFITSYS/include" );
  .x Analysis_Step6.C++("Final", "", "", "", -1, -1, -1, "");
//  .x Analysis_Step6.C+ ("Final", "", "", "", -1, -1, -1, "_SystP");
//  .x Analysis_Step6.C+ ("Final", "", "", "", -1, -1, -1, "_SystI");
//  .x Analysis_Step6.C+ ("Final", "", "", "", -1, -1, -1, "_SystM");
//  .x Analysis_Step6.C+ ("Final", "", "", "", -1, -1, -1, "_SystT");
EOF


root -l -b << EOF
  TString makeshared(gSystem->GetMakeSharedLib());
  TString dummy = makeshared.ReplaceAll("-W ", "");
  TString dummy = makeshared.ReplaceAll("-Wshadow ", "");
  gSystem->SetMakeSharedLib(makeshared);
  gSystem->SetIncludePath("-I$ROOFITSYS/include");
  .x Analysis_Step6.C++("tmp");
EOF


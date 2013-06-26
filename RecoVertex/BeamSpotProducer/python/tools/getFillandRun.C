//
// File: ConditionBrowser_1279914969655.C
//
// run this script
// root -b -q ConditionBrowser_1279914969655.C
//

char NAME[200];
Int_t NROW;
Double_t WEIGHT[20000];
Int_t TIME[20000];
Double_t VALUE[20000];

void getFillandRun()
{
  TFile f("ConditionBrowser_1279914969655.root");
  TTree *t = (TTree*)f.Get("tree");
  t->SetBranchAddress("NAME", &NAME);
  t->SetBranchAddress("NROW", &NROW);
  t->SetBranchAddress("WEIGHT", WEIGHT);
  t->SetBranchAddress("TIME", TIME);
  t->SetBranchAddress("VALUE", VALUE);
  Int_t nentries = (Int_t)t->GetEntries();

  std::map< int, int> fillmap;
  std::map< int, int> runmap;

 
  for (Int_t i=0; i<nentries; i++)
  {
    t->GetEntry(i);
    cout << NAME << endl;
    cout << NROW << endl;
    //cout << "ROW \tWEIGHT \tTIME \tVALUE" << endl;
    for (Int_t j=0; j<NROW; j++)
    {
      //cout << (j+1) << " \t"
      //     << WEIGHT[j] << " \t"
      //     << SecUTC(TIME[j]) << " \t"
      //     << VALUE[j] << endl;
      if (i==0) {
	fillmap[VALUE[j]] = TIME[j];
      } else {
	runmap[VALUE[j]] = TIME[j];
      }
    }
  }
  
  for (std::map<int,int>::const_iterator it=fillmap.begin(); it != fillmap.end(); ++it ) {
    
    int t0 = it->second;
    int afill = it->first;
    ++it;
    //if ( it == fillmap.end() ) break;
    int t1 = it->second;
    --it;

    int run0 = 0;
    int run1 = 0;

    for (std::map<int,int>::const_iterator itt=runmap.begin(); itt != runmap.end(); ++itt) {
      
      int t0run = itt->second;
      int run = itt->first;

      if ( t0run>= t0 && t0run< t1 ) {
	if ( run0==0 ) run0 = run;
	run1 = run;
      }
    
    }
    cout << "fill: "<< afill << " run: " << run0 << " - " << run1 << endl; 
  }
  f.Close();
}

TString SecUTC(Int_t sec)
{
  TTimeStamp ts(sec, 0);
  TString s = ts.AsString("c");
  return s(0, 4) + "." +
         s(5, 2) + "." +
         s(8, 2) + " " +
         s(11, 2) + ":" +
         s(14, 2) + ":" +
         s(17, 2);
}

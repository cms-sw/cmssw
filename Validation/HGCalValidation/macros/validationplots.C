//Author: Andreas Psallidas
//A very helpful small macro to easily make plots of the final root file
void plotkeys(TDirectory *curdir, TFile *newfi, TString dir, string input, string objecttoplot, string sample) {
  curdir = gDirectory;
  TIter next(curdir->GetListOfKeys());
  TKey *key;
  TCanvas c1;

  newfi->mkdir(dir);
  newfi->cd(dir);

  while ((key = (TKey *)next())) {
    key->ls();
    TClass *cl = gROOT->GetClass(key->GetClassName());
    TH1 *h = (TH1 *)key->ReadObj();
    h->Draw();
    // std::cout<< key->GetName() << std::endl;

    TString plotname;
    TString keyname = key->GetName();
    plotname = (keyname + ".png");
    if (sample == "") {
      c1.SaveAs(dir + '/' + plotname);
    } else {
      if (objecttoplot == "Calibrations") {
        c1.SaveAs(sample + '/' + plotname);
      }
      if (objecttoplot == "CaloParticles") {
        //unsigned last = dir.find_last_of("/");
        unsigned last = dir.Last('/');
        c1.SaveAs(sample + '/' + dir(last + 1, dir.Length()) + '/' + plotname);
      }
    }
    h->Write();
  }
}

void validationplots(string input, string objecttoplot, string sample = "") {
  gROOT->ForceStyle();
  gStyle->SetOptStat("ksiourmen");

  TFile *file = TFile::Open(input.c_str());
  TDirectory *currentdir;
  //Folder to be examined and plot objects
  std::vector<TString> folders;
  if (objecttoplot == "SimHits") {
    folders.push_back("hgcalSimHitStudy");
  } else if (objecttoplot == "Digis") {
    folders.push_back("hgcalDigiStudyEE");
    folders.push_back("hgcalDigiStudyHEF");
    folders.push_back("hgcalDigiStudyHEB");
  } else if (objecttoplot == "RecHits") {
    folders.push_back("hgcalRecHitStudyEE");
    folders.push_back("hgcalRecHitStudyHEF");
    folders.push_back("hgcalRecHitStudyHEB");
  } else if (objecttoplot == "CaloParticles") {
    folders.push_back("DQMData/Run 1/HGCAL/Run summary/CaloParticles/-11");
    folders.push_back("DQMData/Run 1/HGCAL/Run summary/CaloParticles/-13");
    folders.push_back("DQMData/Run 1/HGCAL/Run summary/CaloParticles/-211");
    folders.push_back("DQMData/Run 1/HGCAL/Run summary/CaloParticles/-321");
    folders.push_back("DQMData/Run 1/HGCAL/Run summary/CaloParticles/11");
    folders.push_back("DQMData/Run 1/HGCAL/Run summary/CaloParticles/111");
    folders.push_back("DQMData/Run 1/HGCAL/Run summary/CaloParticles/13");
    folders.push_back("DQMData/Run 1/HGCAL/Run summary/CaloParticles/211");
    folders.push_back("DQMData/Run 1/HGCAL/Run summary/CaloParticles/22");
    folders.push_back("DQMData/Run 1/HGCAL/Run summary/CaloParticles/321");
  } else if (objecttoplot == "Calibrations") {
    folders.push_back("DQMData/Run 1/HGCalHitCalibration/Run summary");
  }

  TFile *newfi = new TFile("newfi.root", "recreate");

  for (std::vector<TString>::iterator fol = folders.begin(); fol != folders.end(); ++fol) {
    file->cd((*fol));
    plotkeys(currentdir, newfi, (*fol), input, objecttoplot, sample);
  }

  newfi->Close();
}

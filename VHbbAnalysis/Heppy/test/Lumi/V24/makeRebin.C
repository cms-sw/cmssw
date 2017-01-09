void makeRebin(TString namefile, int lastbin=32){

TFile * output = new TFile(namefile,"RECREATE");
TFile * input = new TFile(("original/"+namefile));

input->cd();

TH1D* pileUp = (TH1D*)input->Get("pileup");
TH1D*  hpileUp = new TH1D("pileup","pileup",lastbin,0,lastbin);

double mc=0;
for (int i=0; i<=50; i++) {

	if(i<lastbin){
		hpileUp->SetBinContent(i,pileUp->GetBinContent(i));
	}
	else{
		mc=mc+pileUp->GetBinContent(i);

	}


}

hpileUp->SetBinContent(lastbin,mc);

output->cd();
hpileUp->Write();
output->Write();
output->Close();


}

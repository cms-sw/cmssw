#include "Riostream.h"
#include <iostream>
#include <map>
#include <sstream>
#include "TVectorF.h"
#include "TSystem.h"
#include "TInterpreter.h"
#include "TF1.h"
#include <TChain.h>
#include <TFileSet.h>
#include <TList.h>
#include <TRandom2.h>
#include "toolbox/toolbox.C"
#include "TLegend.h"
#include "toolbox/L1GtNtuple.h"
#include "hist.C"
#include "Style.C"
#include <stdio.h>
#include <stdlib.h>
#include <sstream>
#include <string>

using namespace std;

#ifndef L1PrescalesSimulator_h
#define L1PrescalesSimulator_h

class L1PrescalesSimulator: public L1GtNtuple {
public:

	//constructor    
	L1PrescalesSimulator(std::string filename) :
		L1GtNtuple(filename) {
          rndng_ = new TRandom2();
          rndng_->SetSeed();
	}
	L1PrescalesSimulator() {
	}

	void run(int nevs=-1);

	~L1PrescalesSimulator() {
	}

protected:

	// these routines are shared with rates analyzer

	void prescale(ULong64_t& word, const unsigned int bit, const float factor);

	void getPrescales();

	// map to define the prescaling
	std::map<unsigned int, float> bit2prescale_;
	
	TRandom *rndng_;

private:

	void loop(const std::map<unsigned int, float>& bit2prescale_);
	void getLuminosities();
	void addGraphsForSimulatedNOfBx();

	toolbox tb_;

	// definitions
	double lumiSectTimeNs_;

	// luminosity range relevant for fit
	Axis_t xFitMin_;
	Axis_t xFitMax_;

	// luminosity range on x axis
	float xMin_;
	float xMax_;

	// constants
	double commonPrescale_;

	int numberOfBunches_;

	int nOfPlots_;

	// buffers

	// the first key of the following maps is the run number
	map<int, map<int, float> > lumiSec2Lumi_;
	map<int, map<int, float> > lumiSec2rate_;
	map<int, map<int, float> > lumiSec2ratePrescaled_;

	std::vector<TVectorF> rateVecs_;

	TVectorF lumiVec_;

	// multiplicators for number of bxs (for each a seperate graph will be drawn

	vector<float> bxMultiplicatorVec_;

	// stores parameters
	std::map<string, string> parameterMap_;

	int yMax_;

	// contains masks for the trigger bits
	// ind0: bits 0-63
	// ind1: bits 63-127
	// ind1: bits 128-192
	vector<ULong64_t> bitMaskVec_;

	TLegend* l_;

	TH1D* Hbits_;
	TH1D* Hbits2_;


};

#endif

void L1PrescalesSimulator::getPrescales() {
	bit2prescale_.clear();

	std::vector<string> file = tb_.readFile(tb_.getMacroDir()
			+ "conf/prescales.dat");

	cout << "Prescales:" << endl;

	for (unsigned int i = 0; i < file.size(); i++) {

		string str = file.at(i);

		std::vector<string> buffer;

		int pos = str.find("=");
		if (pos != -1) {
			string bit, prescale;
			bit = str.substr(0, pos);
			prescale = str.substr(pos + 1, str.length() - pos - 1);

			const unsigned int bitInt=toolbox::convertFromString<unsigned int>(bit);
			const float prescFloat=toolbox::convertFromString<float>(prescale);

			if(bitInt>191) //Algorithm Triggers 0-127, Technical Triggers 128-191
				throw std::runtime_error("A Trigger bit > 191 was given in the prescales configuration; that is not allowed.");

			cout << bitInt << ":" << prescFloat << endl;

			bit2prescale_[bitInt] = prescFloat;

		}
	}

}

void L1PrescalesSimulator::run(int nevs) {

	cout << "Start..." << endl;

	lumiSectTimeNs_ = 23.31;

	l_ = new TLegend(0.6, 0.7, 0.85, 0.9);

	Hbits_ = new TH1D("Hbits","",192,0.,192.);
	Hbits2_ = new TH1D("Hbits2","",192,0.,192.);

	gStyle->SetOptTitle(kFALSE);
	Hbits_->SetTitle("");
	Hbits_->GetXaxis()->SetTitle("Bit");
	Hbits2_->SetTitle("");
	Hbits2_->GetXaxis()->SetTitle("Bit");

	// clear buffers
	lumiSec2Lumi_.clear();
	lumiSec2rate_.clear();
	lumiSec2ratePrescaled_.clear();
	rateVecs_.clear();
	bit2prescale_.clear();
	bxMultiplicatorVec_.clear();
	lumiVec_.ResizeTo(0);
	bitMaskVec_.clear();

	cout << "Open config file: " << tb_.getMacroDir()
			<< "conf/PrescalesSimulator.conf" << endl;

	tb_.readConfigFile(tb_.getMacroDir() + "conf/PrescalesSimulator.conf",
			parameterMap_);

	// fill parameters
	numberOfBunches_ = toolbox::convertFromString<int>(parameterMap_["nOfBunches"]);
	commonPrescale_ = toolbox::convertFromString<int>(parameterMap_["commonPrescale"]);
	xMin_ = toolbox::convertFromString<float>(parameterMap_["xMin"]);
	xMax_ = toolbox::convertFromString<float>(parameterMap_["xMax"]);
	xFitMin_ = toolbox::convertFromString<float>(parameterMap_["xFitMin"]);
	xFitMax_ = toolbox::convertFromString<float>(parameterMap_["xFitMax"]);
	yMax_ = toolbox::convertFromString<int>(parameterMap_["yMax"]);

	int tmp = 1;

	std::map<string, string>::iterator itr;

	do {
		ostringstream oss;

		oss << tmp;

		itr = parameterMap_.find(oss.str());

		if (itr != parameterMap_.end())

			bxMultiplicatorVec_.push_back(toolbox::convertFromString<float>(itr->second));

		tmp++;

	} while (itr != parameterMap_.end());

	tmp = 1;

	tb_.readConfigFile(tb_.getMacroDir() + "conf/bitmasks.dat", parameterMap_);

	cout << "Bit masks: " << endl;

	do {
		ostringstream oss;

		oss << "m" << tmp;

		itr = parameterMap_.find(oss.str());

		if (itr != parameterMap_.end()) {
			cout << (itr->second) << endl;

			bitMaskVec_.push_back(toolbox::convertFromString<ULong64_t>(itr->second, 16));

		}

		tmp++;

	} while (itr != parameterMap_.end());

	// original luminosity and prescaled luminosity + bx multiplicator
	nOfPlots_ = 2 + bxMultiplicatorVec_.size();

	if (nevs) {

		hreset();

		getPrescales();

		getLuminosities();

		loop(bit2prescale_);

	}


	TCanvas* c2 = new TCanvas("c2", "", 900, 700);

	c2->GetEvent();

	Hbits_->Draw();
	Hbits2_->SetLineColor(2);
	Hbits2_->Draw("same");

	// draw the graphs  	

	TCanvas* c1 = new TCanvas("c1", "", 900, 700);
	c1->SetGrid(1, 1);

	TGraph* g = new TGraph(lumiVec_, rateVecs_[0]);

	g->GetXaxis()->SetLimits(xMin_, xMax_);

	g->GetXaxis()->SetTitle("Luminosity per bunch [10^{30} Hz/cm^{2}]");
	g->GetYaxis()->SetTitle("Rate [Hz]");

	g->Fit("pol2", "", "", xFitMin_, xFitMax_);

	g->SetMaximum(yMax_);

	g->Draw("A*");

	gStyle->SetOptTitle(kFALSE);
	g->SetTitle();

	ostringstream buf, buf2;
	buf << numberOfBunches_ << " (original rate)";
	buf2 << numberOfBunches_ << "b";

	l_->AddEntry(g, tb_.toCStr(buf), "l");

	TGraph* g2 = new TGraph(lumiVec_, rateVecs_[1]);

	g2->SetMarkerStyle(21);
	g2->SetLineColor(2);

	g2->Fit("pol2", "", "", xFitMin_, xFitMax_);

	TF1* fitFkt = g2->GetFunction("pol2");
	fitFkt->SetLineColor(2);

	l_->AddEntry(g2, tb_.toCStr(buf2), "l");

	g2->Draw("*");

	addGraphsForSimulatedNOfBx();

	l_->Draw();


}

////////////////////////////////////////////////////////////////////////////////////////////

void L1PrescalesSimulator::addGraphsForSimulatedNOfBx() {
	for (unsigned int r = 0; r < bxMultiplicatorVec_.size(); r++) {

		TF1* fitFkt;

		TGraph* g3 = new TGraph(lumiVec_, rateVecs_[2 + r]);

		g3->SetMarkerStyle(21);
		g3->SetLineColor(tb_.getColor(r + 3));

		g3->Fit("pol2", "", "", xFitMin_, xFitMax_);
		fitFkt = g3->GetFunction("pol2");
		fitFkt->SetLineColor((r + 3));

		ostringstream buf;
		buf << bxMultiplicatorVec_[r] << "b";
		l_->AddEntry(g3, tb_.toCStr(buf), "l");

		g3->Draw("*");
	}

}


void  L1PrescalesSimulator::getLuminosities() {

	TString lumiDir;
	lumiDir.Append(directory_+"/lumis/lumis.root");

	TFile f(lumiDir.Data());

	TTree* const tree=dynamic_cast<TTree*>(f.Get("ntuple"));
	if(!tree)
		throw std::runtime_error("Expected object \"ntuple\" in luminosities file to be of type TTree, which is was not.");

	//Float_t run,ls,lumiDelivered, lumiReported;

	float run;
	float ls;
	float lumiReported,lumiDelivered;

	// Linking the local variables to the tree branches
	tree->SetBranchAddress("run", &run);
	tree->SetBranchAddress("ls", &ls);
	tree->SetBranchAddress("lumiDelivered", &lumiDelivered);
	tree->SetBranchAddress("lumiReported", &lumiReported);

	Long64_t nEntries = tree->GetEntries();

	for (Long64_t iEnt=0; iEnt<nEntries; ++iEnt)
	{
		tree->GetEntry(iEnt); 	
		lumiSec2Lumi_[run][ls]=lumiReported;
	}

}


void L1PrescalesSimulator::loop(const std::map<unsigned int, float>& bit2prescale_)
{

	Long64_t nevents(-1);

	//number of events to process
	if (nevents == -1 || nevents > GetEntries())
		nevents = GetEntries();
	std::cout << nevents << " to process ..." << std::endl;

	//loop over the events
	for (Long64_t i = 0; i < nevents; i++) {
		//load the i-th event 
		Long64_t ientry = LoadTree(i);
		if (ientry < 0)
			break;
		GetEntry(i);

		ULong64_t a1 = gt_->tw1.at(2) & bitMaskVec_[0];
		ULong64_t a2 = gt_->tw2.at(2) & bitMaskVec_[1];
		ULong64_t tt = gt_->tt.at(2) & bitMaskVec_[2];
		const int run = event_->run;

		const int ls = event_->lumi;

		if (a1 || a2 || tt) {

			lumiSec2rate_[run][ls]++;

			for(int ibit=0; ibit<64; ibit++) {
                           if( (a1>>ibit)&1 ) Hbits_->Fill(float(ibit));
                           if( (a2>>ibit)&1 ) Hbits_->Fill(float(ibit+64));
                           if( (tt>>ibit)&1 ) Hbits_->Fill(float(ibit+128));
                        }

		}

		//process progress
		if (i != 0 && (i % 10000) == 0) {
			std::cout << "- processing event " << i << ", Run: " << run
					<< " ,tw2: " << a2 << " ,LS: " << ls << "\r" << std::flush;
		}

		for (std::map<unsigned int, float>::const_iterator it = bit2prescale_.begin(); it
				!= bit2prescale_.end(); it++) {
			const unsigned int bit = it->first;
			const float presc = it->second;

			if (presc != -1.)
			{
				if(bit < 64)
				{
					prescale(a1, bit, presc);
				}
				else if(bit < 128)
				{
					prescale(a2, bit-64, presc);
				}
				else
				{
					prescale(tt, bit-128, presc);
				}
			}

		}

		if (a1 || a2 || tt) {

			lumiSec2ratePrescaled_[run][event_->lumi]++;
			for(int ibit=0; ibit<64; ibit++) {
                           if( (a1>>ibit)&1 ) Hbits2_->Fill(float(ibit));
                           if( (a2>>ibit)&1 ) Hbits2_->Fill(float(ibit+64));
                           if( (tt>>ibit)&1 ) Hbits2_->Fill(float(ibit+128));
                        }
		}

	}

	// loop over runs

	for (map<int, map<int, float> >::iterator itr = lumiSec2Lumi_.begin(); itr
			!= lumiSec2Lumi_.end(); itr++) {

		map<int, float> &ls2lumi = itr->second;
		const int run = itr->first;

		cout << "Run: " << run << endl;

		// the number of lumi sections to loop over
		const unsigned int nOfLumiSections = ls2lumi.size();

		lumiVec_.ResizeTo(nOfLumiSections);

		rateVecs_.resize(nOfPlots_);

		for (int k = 0; k < nOfPlots_; k++)
			rateVecs_[k].ResizeTo(nOfLumiSections);

		for (unsigned int i = 0; i < nOfLumiSections; i++)
		{
			if (!(lumiSec2rate_[run][i] == 0))
			{
				lumiVec_[i] = ls2lumi[i] / float(numberOfBunches_);
				rateVecs_[0][i] = (lumiSec2rate_[run][i] * commonPrescale_
						/ lumiSectTimeNs_);
				rateVecs_[1][i] = (lumiSec2ratePrescaled_[run][i]
						* commonPrescale_ / lumiSectTimeNs_);

				for (unsigned int r = 0; r < bxMultiplicatorVec_.size(); r++)
				{
					rateVecs_[r + 2][i] = (lumiSec2ratePrescaled_[run][i]
							* commonPrescale_ / lumiSectTimeNs_
							/ float(numberOfBunches_)
							* bxMultiplicatorVec_[r]);
					rateVecs_[r + 2][i] = (lumiSec2ratePrescaled_[run][i]
							* commonPrescale_ / lumiSectTimeNs_
							/ float(numberOfBunches_)
							* bxMultiplicatorVec_[r]);
				}
			}
		}
	}

	cout
			<< "                                                                        "
			<< std::endl;
	return;
}


void L1PrescalesSimulator::prescale(ULong64_t& word, const unsigned int bit, const float factor) {

	if ((word >> bit) & 1) {
		if(rndng_->Rndm() > 1./factor ) {
			word &= ~(1ULL << bit);
		}
	}
}


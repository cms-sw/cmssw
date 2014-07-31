#include "TString.h"
#include "TObject.h"
#include "TFile.h"
#include "TVector.h"
#include "TGraph.h"
#include "TF1.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <map>

#define MIN_EXTRAP_PU 16
#define MAX_EXTRAP_PU 28

class extrapolateBit {

	public :

	extrapolateBit(TString name, Int_t bit, TVectorF x, TVectorF y, Float_t ratePU28, Float_t ratePU34,TString fitfunction): name_(name) , bit_(bit) , x_(x) , y_(y) , ratePU28_(ratePU28), ratePU34_(ratePU34), fitfunction_(fitfunction){
		graph_ = new TGraph(x_,y_);
		graph_->SetName(TString("g_"+name_).Data());
		graph_->SetTitle(TString(name_).Data());

		cout << fitfunction << endl;

		extrapFunction_ = new TF1(TString("f_"+name_).Data(),fitfunction,MIN_EXTRAP_PU,MAX_EXTRAP_PU);
		extrapFunction_->SetLineColor(kRed);
	}
	~extrapolateBit() {};

	Float_t extrapolate() {
		graph_->Fit(extrapFunction_,"RQ"); // CB Fit Options

		//std::cout << name_.Data() << "\t" << extrapFunction_->Eval(MAX_EXTRAP_PU) << "\t" << ratePU34_ << "/"<< ratePU28_ << std::endl; 
		
		Float_t scalefrom7TeV = 1;
		if ((ratePU28_ > 0.0001) && (ratePU34_ > 0.0001)) {
			scalefrom7TeV = ratePU34_/ratePU28_;
		}
		
		return extrapFunction_->Eval(MAX_EXTRAP_PU) * scalefrom7TeV;
	};

	TGraph * getGraph() { return graph_; };

	private :

	TString name_;
	Int_t bit_;

	TVectorF x_;
	TVectorF y_;

	Float_t ratePU28_;
	Float_t ratePU34_;

	TString fitfunction_;

	TGraph *graph_;
	TF1* extrapFunction_;

};


class extrapolateMenu {

	public :
	extrapolateMenu();
	~extrapolateMenu() { };

	private :
	void readScaleFile(TString scaleFile, float pileUp);
	void readRateFile(TString rateFile, float pileUp);

	std::map<Int_t,TString> bitToName;
	std::map<Float_t,std::map<TString,Float_t> > readFileMap;
	std::map<Float_t,std::map<TString,Float_t> > scaleFileMap;
	std::map<std::string, int> BitMapping;
	std::map<std::string, std::string> FitFunction;

	void MyInit() {
		BitMapping["L1_ZeroBias"] = 0;
		BitMapping["L1_ZeroBias_Instance1"] = 1;
		BitMapping["L1_BeamGas_Hf_BptxPlusPostQuiet"] = 2;
		BitMapping["L1_BeamGas_Hf_BptxMinusPostQuiet"] = 4;
		BitMapping["L1_InterBunch_Bptx"] = 5;
		BitMapping["L1_BeamHalo"] = 8;
		BitMapping["L1_TripleMu0"] = 9;
		BitMapping["L1_Mu4_HTT125"] = 10;
		BitMapping["L1_Mu3p5_EG12"] = 11;
		BitMapping["L1_Mu12er_ETM20"] = 12;
		BitMapping["L1_MuOpen_EG12"] = 13;
		BitMapping["L1_Mu12_EG7"] = 14;
		BitMapping["L1_SingleJet16"] = 15;
		BitMapping["L1_SingleJet36"] = 16;
		BitMapping["L1_SingleJet52"] = 17;
		BitMapping["L1_SingleJet68"] = 18;
		BitMapping["L1_SingleJet92"] = 19;
		BitMapping["L1_SingleJet128"] = 20;
		BitMapping["L1_DoubleEG6_HTT100"] = 21;
		BitMapping["L1_DoubleEG6_HTT125"] = 22;
		BitMapping["L1_Mu5_DoubleEG5"] = 23;
		BitMapping["L1_DoubleMu3p5_EG5"] = 24;
		BitMapping["L1_DoubleMu5_EG5"] = 25;
		BitMapping["L1_DoubleMu0er_HighQ"] = 26;
		BitMapping["L1_Mu5_DoubleEG6"] = 27;
		BitMapping["L1_DoubleJetC44_ETM30"] = 28;
		BitMapping["L1_Mu3_JetC16_WdEtaPhi2"] = 29;
		BitMapping["L1_Mu3_JetC52_WdEtaPhi2"] = 30;
		BitMapping["L1_SingleEG7"] = 31;
		BitMapping["L1_SingleIsoEG20er"] = 32;
		BitMapping["L1_EG22_ForJet24"] = 33;
		BitMapping["L1_EG22_ForJet32"] = 34;
		BitMapping["L1_DoubleJetC44_Eta1p74_WdEta4"] = 35;
		BitMapping["L1_DoubleJetC56_Eta1p74_WdEta4"] = 36;
		BitMapping["L1_DoubleTauJet44er"] = 37;
		BitMapping["L1_DoubleEG_13_7"] = 38;
		BitMapping["L1_TripleEG_12_7_5"] = 39;
		BitMapping["L1_HTT125"] = 40;
		BitMapping["L1_DoubleJetC52"] = 41;
		BitMapping["L1_SingleMu14er"] = 42;
		BitMapping["L1_SingleIsoEG18er"] = 43;
		BitMapping["L1_DoubleMu_10_Open"] = 44;
		BitMapping["L1_DoubleMu_10_3p5"] = 45;
		BitMapping["L1_ETT80"] = 46;
		BitMapping["L1_SingleEG5"] = 47;
		BitMapping["L1_SingleEG18er"] = 48;
		BitMapping["L1_SingleEG22"] = 49;
		BitMapping["L1_SingleEG12"] = 50;
		BitMapping["L1_SingleEG24"] = 51;
		BitMapping["L1_SingleEG20"] = 52;
		BitMapping["L1_SingleEG30"] = 53;
		BitMapping["L1_DoubleMu3er_HighQ_WdEta22"] = 54;
		BitMapping["L1_SingleMuOpen"] = 55;
		BitMapping["L1_SingleMu16"] = 56;
		BitMapping["L1_SingleMu3"] = 57;
		BitMapping["L1_DoubleMu_5er_0er_HighQ_WdEta22"] = 58;
		BitMapping["L1_SingleMu7"] = 59;
		BitMapping["L1_SingleMu20er"] = 60;
		BitMapping["L1_SingleMu12"] = 61;
		BitMapping["L1_SingleMu20"] = 62;
		BitMapping["L1_SingleMu25er"] = 63;
		BitMapping["L1_ETM100"] = 64;
		BitMapping["L1_ETM36"] = 65;
		BitMapping["L1_ETM30"] = 66;
		BitMapping["L1_ETM50"] = 67;
		BitMapping["L1_ETM70"] = 68;
		BitMapping["L1_ETT300"] = 69;
		BitMapping["L1_HTT100"] = 70;
		BitMapping["L1_HTT150"] = 71;
		BitMapping["L1_HTT175"] = 72;
		BitMapping["L1_HTT200"] = 73;
		BitMapping["L1_Mu10er_JetC12_WdEtaPhi1_DoubleJetC_20_12"] = 74;
		BitMapping["L1_Mu10er_JetC32"] = 75;
		BitMapping["L1_DoubleJetC64"] = 76;
		BitMapping["L1_Mu10er_JetC12_WdEtaPhi1_DoubleJetC_32_12"] = 77;
		BitMapping["L1_SingleJetC32_NotBptxOR"] = 78;
		BitMapping["L1_ETM40"] = 79;
		BitMapping["L1_Mu0_HTT50"] = 80;
		BitMapping["L1_Mu0_HTT100"] = 81;
		BitMapping["L1_DoubleEG5"] = 82;
		BitMapping["L1_IsoEG18er_JetC_Cen36_Tau28_dPhi1"] = 83;
		BitMapping["L1_EG18er_JetC_Cen36_Tau28_dPhi1"] = 84;
		BitMapping["L1_SingleMu16er"] = 86;
		BitMapping["L1_EG18er_JetC_Cen28_Tau20_dPhi1"] = 87;
		BitMapping["L1_IsoEG18er_JetC_Cen32_Tau24_dPhi1"] = 88;
		BitMapping["L1_SingleMu6_NotBptxOR"] = 89;
		BitMapping["L1_Mu8_DoubleJetC20"] = 90;
		BitMapping["L1_DoubleMu0"] = 92;
		BitMapping["L1_EG8_DoubleJetC20"] = 94;
		BitMapping["L1_DoubleMu5"] = 95;
		BitMapping["L1_DoubleJetC56"] = 96;
		BitMapping["L1_TripleMu0_HighQ"] = 97;
		BitMapping["L1_TripleMu_5_5_0"] = 98;
		BitMapping["L1_ETT140"] = 99;
		BitMapping["L1_DoubleJetC36"] = 100;
		BitMapping["L1_DoubleJetC36_ETM30"] = 101;
		BitMapping["L1_SingleJet36_FwdVeto5"] = 102;
		BitMapping["L1_TripleJet_64_44_24_VBF"] = 103;
		BitMapping["L1_TripleJet_64_48_28_VBF"] = 104;
		BitMapping["L1_TripleJet_68_48_32_VBF"] = 105;
		BitMapping["L1_QuadJetC40"] = 106;
		BitMapping["L1_QuadJetC36"] = 107;
		BitMapping["L1_TripleJetC_52_28_28"] = 108;
		BitMapping["L1_QuadJetC32"] = 109;
		BitMapping["L1_DoubleForJet16_EtaOpp"] = 110;
		BitMapping["L1_DoubleEG3_FwdVeto"] = 111;
		BitMapping["L1_SingleJet20_Central_NotBptxOR"] = 112;
		BitMapping["L1_SingleJet16_FwdVeto5"] = 113;
		BitMapping["L1_SingleForJet16"] = 114;
		BitMapping["L1_DoubleJetC36_RomanPotsOR"] = 115;
		BitMapping["L1_SingleMu20_RomanPotsOR"] = 116;
		BitMapping["L1_SingleEG20_RomanPotsOR"] = 117;
		BitMapping["L1_DoubleMu5_RomanPotsOR"] = 118;
		BitMapping["L1_DoubleEG5_RomanPotsOR"] = 119;
		BitMapping["L1_SingleJet52_RomanPotsOR"] = 120;
		BitMapping["L1_SingleMu18er"] = 122;
		BitMapping["L1_MuOpen_EG5"] = 123;
		BitMapping["L1_DoubleMu_12_5"] = 124;
		BitMapping["L1_TripleEG7"] = 125;
		
		FitFunction["L1_ZeroBias"] = "pol2";
		FitFunction["L1_ZeroBias_Instance1"] = "pol2";
		FitFunction["L1_BeamGas_Hf_BptxPlusPostQuiet"] = "pol2";
		FitFunction["L1_BeamGas_Hf_BptxMinusPostQuiet"] = "pol2";
		FitFunction["L1_InterBunch_Bptx"] = "pol2";
		FitFunction["L1_BeamHalo"] = "pol2";
		FitFunction["L1_TripleMu0"] = "pol1";
		FitFunction["L1_Mu4_HTT125"] = "pol2";
		FitFunction["L1_Mu3p5_EG12"] = "pol2";
		FitFunction["L1_Mu12er_ETM20"] = "pol2";
		FitFunction["L1_MuOpen_EG12"] = "pol2";
		FitFunction["L1_Mu12_EG7"] = "pol2";
		FitFunction["L1_SingleJet16"] = "pol2";
		FitFunction["L1_SingleJet36"] = "pol2";
		FitFunction["L1_SingleJet52"] = "pol2";
		FitFunction["L1_SingleJet68"] = "pol2";
		FitFunction["L1_SingleJet92"] = "pol2";
		FitFunction["L1_SingleJet128"] = "pol2";
		FitFunction["L1_DoubleEG6_HTT100"] = "pol2";
		FitFunction["L1_DoubleEG6_HTT125"] = "pol2";
		FitFunction["L1_Mu5_DoubleEG5"] = "pol2";
		FitFunction["L1_DoubleMu3p5_EG5"] = "pol2";
		FitFunction["L1_DoubleMu5_EG5"] = "pol1";
		FitFunction["L1_DoubleMu0er_HighQ"] = "pol1";
		FitFunction["L1_Mu5_DoubleEG6"] = "pol1";
		FitFunction["L1_DoubleJetC44_ETM30"] = "pol2";
		FitFunction["L1_Mu3_JetC16_WdEtaPhi2"] = "pol2";
		FitFunction["L1_Mu3_JetC52_WdEtaPhi2"] = "pol2";
		FitFunction["L1_SingleEG7"] = "pol1";
		FitFunction["L1_SingleIsoEG20er"] = "pol1";
		FitFunction["L1_EG22_ForJet24"] = "pol2";
		FitFunction["L1_EG22_ForJet32"] = "pol2";
		FitFunction["L1_DoubleJetC44_Eta1p74_WdEta4"] = "pol2";
		FitFunction["L1_DoubleJetC56_Eta1p74_WdEta4"] = "pol2";
		FitFunction["L1_DoubleTauJet44er"] = "pol1";
		FitFunction["L1_DoubleEG_13_7"] = "pol1";
		FitFunction["L1_TripleEG_12_7_5"] = "pol1";
		FitFunction["L1_HTT125"] = "pol0";
		FitFunction["L1_DoubleJetC52"] = "pol0";
		FitFunction["L1_SingleMu14er"] = "pol1";
		FitFunction["L1_SingleIsoEG18er"] = "pol1";
		FitFunction["L1_DoubleMu_10_Open"] = "pol1";
		FitFunction["L1_DoubleMu_10_3p5"] = "pol1";
		FitFunction["L1_ETT80"] = "pol2";
		FitFunction["L1_SingleEG5"] = "pol1";
		FitFunction["L1_SingleEG18er"] = "pol1";
		FitFunction["L1_SingleEG22"] = "pol1";
		FitFunction["L1_SingleEG12"] = "pol1";
		FitFunction["L1_SingleEG24"] = "pol1";
		FitFunction["L1_SingleEG20"] = "pol1";
		FitFunction["L1_SingleEG30"] = "pol1";
		FitFunction["L1_DoubleMu3er_HighQ_WdEta22"] = "pol1";
		FitFunction["L1_SingleMuOpen"] = "pol1";
		FitFunction["L1_SingleMu16"] = "pol1";
		FitFunction["L1_SingleMu3"] = "pol0";
		FitFunction["L1_DoubleMu_5er_0er_HighQ_WdEta22"] = "pol1";
		FitFunction["L1_SingleMu7"] = "pol1";
		FitFunction["L1_SingleMu20er"] = "pol1";
		FitFunction["L1_SingleMu12"] = "pol1";
		FitFunction["L1_SingleMu20"] = "pol1";
		FitFunction["L1_SingleMu25er"] = "pol1";
		FitFunction["L1_ETM100"] = "pol2";
		FitFunction["L1_ETM36"] = "pol2";
		FitFunction["L1_ETM30"] = "pol2";
		FitFunction["L1_ETM50"] = "pol2";
		FitFunction["L1_ETM70"] = "pol2";
		FitFunction["L1_ETT300"] = "pol2";
		FitFunction["L1_HTT100"] = "pol0";
		FitFunction["L1_HTT150"] = "pol2";
		FitFunction["L1_HTT175"] = "pol2";
		FitFunction["L1_HTT200"] = "pol2";
		FitFunction["L1_Mu10er_JetC12_WdEtaPhi1_DoubleJetC_20_12"] = "pol2";
		FitFunction["L1_Mu10er_JetC32"] = "pol2";
		FitFunction["L1_DoubleJetC64"] = "pol2";
		FitFunction["L1_Mu10er_JetC12_WdEtaPhi1_DoubleJetC_32_12"] = "pol2";
		FitFunction["L1_SingleJetC32_NotBptxOR"] = "pol2";
		FitFunction["L1_ETM40"] = "pol2";
		FitFunction["L1_Mu0_HTT50"] = "pol0";
		FitFunction["L1_Mu0_HTT100"] = "pol2";
		FitFunction["L1_DoubleEG5"] = "pol2";
		FitFunction["L1_IsoEG18er_JetC_Cen36_Tau28_dPhi1"] = "pol2";
		FitFunction["L1_EG18er_JetC_Cen36_Tau28_dPhi1"] = "pol2";
		FitFunction["L1_SingleMu16er"] = "pol1";
		FitFunction["L1_EG18er_JetC_Cen28_Tau20_dPhi1"] = "pol2";
		FitFunction["L1_IsoEG18er_JetC_Cen32_Tau24_dPhi1"] = "pol2";
		FitFunction["L1_SingleMu6_NotBptxOR"] = "pol2";
		FitFunction["L1_Mu8_DoubleJetC20"] = "pol2";
		FitFunction["L1_DoubleMu0"] = "pol1";
		FitFunction["L1_EG8_DoubleJetC20"] = "pol2";
		FitFunction["L1_DoubleMu5"] = "pol1";
		FitFunction["L1_DoubleJetC56"] = "pol2";
		FitFunction["L1_TripleMu0_HighQ"] = "pol1";
		FitFunction["L1_TripleMu_5_5_0"] = "pol1";
		FitFunction["L1_ETT140"] = "pol2";
		FitFunction["L1_DoubleJetC36"] = "pol2";
		FitFunction["L1_DoubleJetC36_ETM30"] = "pol2";
		FitFunction["L1_SingleJet36_FwdVeto5"] = "pol2";
		FitFunction["L1_TripleJet_64_44_24_VBF"] = "pol2";
		FitFunction["L1_TripleJet_64_48_28_VBF"] = "pol2";
		FitFunction["L1_TripleJet_68_48_32_VBF"] = "pol2";
		FitFunction["L1_QuadJetC40"] = "pol2";
		FitFunction["L1_QuadJetC36"] = "pol0";
		FitFunction["L1_TripleJetC_52_28_28"] = "pol2";
		FitFunction["L1_QuadJetC32"] = "pol0";
		FitFunction["L1_DoubleForJet16_EtaOpp"] = "pol2";
		FitFunction["L1_DoubleEG3_FwdVeto"] = "pol2";
		FitFunction["L1_SingleJet20_Central_NotBptxOR"] = "pol2";
		FitFunction["L1_SingleJet16_FwdVeto5"] = "pol2";
		FitFunction["L1_SingleForJet16"] = "pol2";
		FitFunction["L1_DoubleJetC36_RomanPotsOR"] = "pol2";
		FitFunction["L1_SingleMu20_RomanPotsOR"] = "pol2";
		FitFunction["L1_SingleEG20_RomanPotsOR"] = "pol2";
		FitFunction["L1_DoubleMu5_RomanPotsOR"] = "pol2";
		FitFunction["L1_DoubleEG5_RomanPotsOR"] = "pol2";
		FitFunction["L1_SingleJet52_RomanPotsOR"] = "pol2";
		FitFunction["L1_SingleMu18er"] = "pol1";
		FitFunction["L1_MuOpen_EG5"] = "pol1";
		FitFunction["L1_DoubleMu_12_5"] = "pol1";
		FitFunction["L1_TripleEG7"] = "pol1";
	}

	TString L1FitFunction(string l1name) {

		std::map<std::string, std::string>::const_iterator it = FitFunction.find(l1name);
		if (it == FitFunction.end() ) {
			std::cout << " Wrong L1 name, not in FitFunction " << l1name << std::endl;
			return "ERROR";
		}

		return FitFunction[l1name];
	}


};

void extrapolateMenu::readScaleFile(TString scaleFile, float pileUp) {

	std::ifstream file(scaleFile.Data());  

	std::string  dummy;
	std::string  name;
	Float_t rate;
	Int_t bit;

	while (true) {
		file >> bit >> name >> dummy >> rate >> dummy >> dummy >> dummy >> dummy >> dummy >> dummy >> dummy >> dummy;
		if (!file.good()) break;
		//std::cout << "bit : " << bit << " name : " << name << " rate : " << rate << std::endl;
		scaleFileMap[pileUp][name] = rate;
		bitToName[bit] = name;
	}
};

void extrapolateMenu::readRateFile(TString rateFile, float pileUp) {

	std::ifstream file(rateFile.Data());  

	std::string  dummy;
	std::string  name;
	Float_t rate;  
	Int_t bit;

	while (true) {
		file >> bit >> name >> dummy >> rate >> dummy >> dummy >> dummy;
		if (!file.good()) break;
		//std::cout << "bit : " << bit << " name : " << name << " rate : " << rate << std::endl;
		readFileMap[pileUp][name] = rate;
	}
};

extrapolateMenu::extrapolateMenu() {

	MyInit();

	TFile *outFile = new TFile("50_outFile.root","RECREATE");

	readScaleFile("50_28_0.437_rates.txt",28);
	readScaleFile("50_34_0.529_rates.txt",34);

// CB here are files for 2012 rates and their PU
	std::map<float,TString> dataFiles;
//	dataFiles[26] = "50_25.5_54.4649_rates.txt";
	dataFiles[24] = "50_24_47.2923_rates.txt";
	dataFiles[21] = "50_21_44.2939_rates.txt";
	dataFiles[19] = "50_19_41.0814_rates.txt";
//	dataFiles[12] = "50_12_37.5111_rates.txt";

	std::map<float,TString>::const_iterator dataFilesIt  = dataFiles.begin();
	std::map<float,TString>::const_iterator dataFilesEnd = dataFiles.end();

	for(;dataFilesIt!=dataFilesEnd;++dataFilesIt) {
		readRateFile(dataFilesIt->second,dataFilesIt->first);
	}

	std::map<Int_t,extrapolateBit*> triggersToExtrapolate;

	std::map<Int_t,TString>::const_iterator bitToNameIt  = bitToName.begin();
	std::map<Int_t,TString>::const_iterator bitToNameEnd = bitToName.end();

	cout << "a " << bitToName.size() << endl;
	for(;bitToNameIt!=bitToNameEnd;++bitToNameIt) {

		Int_t    bit = bitToNameIt->first;
		TString  triggerName = bitToNameIt->second;

		TVectorF x(dataFiles.size());
		TVectorF y(dataFiles.size());

		std::map<Float_t,std::map<TString,Float_t> >::const_iterator readFileMapIt  = readFileMap.begin();
		std::map<Float_t,std::map<TString,Float_t> >::const_iterator readFileMapEnd = readFileMap.end();

		for(Int_t i=0;readFileMapIt!=readFileMapEnd;++readFileMapIt,++i) {
			x[i] = readFileMapIt->first;
			y[i] = readFileMapIt->second.find(triggerName)->second;
		}     

		triggersToExtrapolate[bit] = new extrapolateBit(triggerName,bit,x,y,scaleFileMap[28][triggerName],scaleFileMap[34][triggerName],FitFunction[(std::string)triggerName]);

	}

	cout << "b" << endl;


	Float_t sumrate = 0;

	bitToNameIt  = bitToName.begin();
	bitToNameEnd = bitToName.end();

	for(;bitToNameIt!=bitToNameEnd;++bitToNameIt) {

		Int_t    bit = bitToNameIt->first;
		TString  triggerName = bitToNameIt->second;

		Float_t rate = triggersToExtrapolate[bit]->extrapolate();
		
		if (rate > 0.001) {
			sumrate +=rate;
		}
		
		cout << bit << "\t" << triggerName.Data() << "\t" << rate << endl;

		outFile->WriteObject(triggersToExtrapolate[bit]->getGraph(),("g_"+triggerName).Data());
	}


	cout << "Combined rate: " << sumrate << endl;

	outFile->Close();

}    





















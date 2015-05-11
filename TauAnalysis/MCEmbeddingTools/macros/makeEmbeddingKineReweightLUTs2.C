
#include <TFile.h>
#include <TString.h>
#include <TH1.h>
#include <TH2.h>
#include <TCanvas.h>
#include <TLegend.h>
#include <TTree.h>
#include <TLorentzVector.h>
#include <TMath.h>
#include <TROOT.h>

#include <vector>
#include <iostream>
#include <iomanip>

enum { kUndefined, kMuon1Pt, kMuon1Eta, kMuon2Pt, kMuon2Eta, kDiMuonPt, kDiMuonMass };

struct weightEntryType
{
  weightEntryType(const TString& name, const TH1* lut, int variableX)
    : name_(name),
      lut_(lut),
      xAxis_(lut->GetXaxis()),
      numBinsX_(lut->GetNbinsX()),
      yAxis_(0),
      numBinsY_(-1),
      variableX_(variableX),            
      variableY_(kUndefined)
  {}
  weightEntryType(const TString& name, const TH2* lut, int variableX, int variableY)
    : name_(name),
      lut_(lut),
      xAxis_(lut->GetXaxis()),
      numBinsX_(lut->GetNbinsX()),
      yAxis_(lut->GetYaxis()),
      numBinsY_(lut->GetNbinsY()),
      variableX_(variableX),            
      variableY_(variableY)
  {}
  Float_t operator()(Float_t genMuon1Pt, Float_t genMuon1Eta, Float_t genMuon2Pt, Float_t genMuon2Eta, Float_t genDiMuonPt, Float_t genDiMuonMass)
  {
    //std::cout << "<weightEntryType::operator()>:" << std::endl;
    //std::cout << " name = " << name_ << std::endl;
    Float_t weight = 1.0;
    if ( yAxis_ ) { // 2d case
      Float_t variableX_value = getVariableValue(genMuon1Pt, genMuon1Eta, genMuon2Pt, genMuon2Eta, genDiMuonPt, genDiMuonMass, variableX_);
      int binX = xAxis_->FindBin(variableX_value);
      if ( binX <= 1 ) binX = 1;
      else if ( binX >= numBinsX_ ) binX = numBinsX_;
      Float_t variableY_value = getVariableValue(genMuon1Pt, genMuon1Eta, genMuon2Pt, genMuon2Eta, genDiMuonPt, genDiMuonMass, variableY_);
      int binY = yAxis_->FindBin(variableY_value);
      if ( binY <= 1 ) binY = 1;
      else if ( binY >= numBinsY_ ) binY = numBinsY_;
      weight = lut_->GetBinContent(binX, binY);
    } else {        // 1d case
      Float_t variableX_value = getVariableValue(genMuon1Pt, genMuon1Eta, genMuon2Pt, genMuon2Eta, genDiMuonPt, genDiMuonMass, variableX_);
      int binX = xAxis_->FindBin(variableX_value);
      if ( binX <= 1 ) binX = 1;
      else if ( binX >= numBinsX_ ) binX = numBinsX_;
      weight = lut_->GetBinContent(binX);
    }
    //std::cout << "--> weight = " << weight << std::endl;
    return weight;
  }
  Float_t getVariableValue(Float_t muon1Pt, Float_t muon1Eta, Float_t muon2Pt, Float_t muon2Eta, Float_t diMuonPt, Float_t diMuonMass, int variable)
  {
    if      ( variable == kMuon1Pt    ) return muon1Pt;
    else if ( variable == kMuon1Eta   ) return muon1Eta;
    else if ( variable == kMuon2Pt    ) return muon2Pt;
    else if ( variable == kMuon2Eta   ) return muon2Eta;
    else if ( variable == kDiMuonPt   ) return diMuonPt;
    else if ( variable == kDiMuonMass ) return diMuonMass;
    else assert(0);
  }
  TString name_;
  const TH1* lut_;
  const TAxis* xAxis_;
  Int_t numBinsX_;
  const TAxis* yAxis_;
  Int_t numBinsY_;
  int variableX_;
  int variableY_;
};

struct plotEntryType
{
  plotEntryType(const std::string& type, const std::string& name)
    : histogramMuon1Pt_varBinning_(0),
      histogramMuon1Pt_fixedBinning_(0),
      histogramMuon1Eta_(0),
      histogramMuon2Pt_varBinning_(0),
      histogramMuon2Pt_fixedBinning_(0),
      histogramMuon2Eta_(0),
      histogramMuon2Pt_vs_muon1Pt_varBinning_(0),
      histogramMuon2Eta_vs_muon1Eta_(0),
      histogramDiMuonPt_varBinning_(0),
      histogramDiMuonPt_fixedBinning_(0),
      histogramDiMuonMass_varBinning_(0),
      histogramDiMuonMass_fixedBinning_(0),
      histogramDiMuonMass_vs_diMuonPt_varBinning_(0),
      histogramNumJetsCorrPtGt20_(0),
      histogramNumJetsCorrPtGt30_(0),
      histogramPFMEt_(0),
      histogramCaloMEt_(0)
  {    
    name_ = Form("%s_%s", type.data(), name.data());
    if      ( type == "Ztautau"     ) type_ = kZtautau;
    else if ( type == "genEmbedded" ) type_ = kGenEmbedded;
    else if ( type == "recEmbedded" ) type_ = kRecEmbedded;
    else if ( type == "genZmumu"    ) type_ = kGenZmumu;
    else if ( type == "recZmumu"    ) type_ = kRecZmumu;
    else {
      std::cerr << "Invalid type = " << type << " !!" << std::endl;
      assert(0);
    }
    std::vector<double> muonPtBinning;
    double muonPt = 0.;
    while ( muonPt <= 250. ) {
      muonPtBinning.push_back(muonPt);
      if      ( muonPt <  20. ) muonPt +=  5.;
      else if ( muonPt <  50. ) muonPt +=  2.;
      else if ( muonPt <  60. ) muonPt +=  5.;
      else if ( muonPt <  80. ) muonPt += 10.;
      else if ( muonPt < 100. ) muonPt += 20.;
      else if ( muonPt < 150. ) muonPt += 50.;
      else                      muonPt += 100.;
    }
    histogramMuon1Pt_varBinning_ = bookHistogram1d_varBinning("muon1Pt_varBinning", name_, muonPtBinning);
    histogramMuon1Pt_fixedBinning_ = bookHistogram1d_fixedBinning("muon1Pt_fixedBinning", name_, 50, 0., 250.);
    histogramMuon1Eta_ = bookHistogram1d_fixedBinning("muon1Eta", name_, 50, -2.5, +2.5);
    histogramMuon2Pt_varBinning_ = bookHistogram1d_varBinning("muon2Pt_varBinning", name_, muonPtBinning);
    histogramMuon2Pt_fixedBinning_ = bookHistogram1d_fixedBinning("muon2Pt_fixedBinning", name_, 50, 0., 250.);
    histogramMuon2Eta_ = bookHistogram1d_fixedBinning("muon2Eta", name_, 50, -2.5, +2.5);
    histogramMuon2Pt_vs_muon1Pt_varBinning_ = bookHistogram2d_varBinning("muon2Pt_vs_muon1Pt_varBinning", name_, muonPtBinning, muonPtBinning);
    histogramMuon2Eta_vs_muon1Eta_ = bookHistogram2d_fixedBinning("muon2Eta_vs_muon1Eta", name_, 25, -2.5, +2.5, 25, -2.5, +2.5);
    std::vector<double> diMuonPtBinning;
    double diMuonPt = 0.;
    while ( diMuonPt <= 250. ) {
      diMuonPtBinning.push_back(diMuonPt);
      if      ( diMuonPt <  20. ) diMuonPt +=   2.;
      else if ( diMuonPt <  40. ) diMuonPt +=   5.;
      else if ( diMuonPt <  50. ) diMuonPt +=  10.;
      else if ( diMuonPt <  75. ) diMuonPt +=  25.;
      else if ( diMuonPt < 150. ) diMuonPt +=  75.;
      else                        diMuonPt += 100.;
    }
    histogramDiMuonPt_varBinning_ = bookHistogram1d_varBinning("diMuonPt_varBinning", name_, diMuonPtBinning);
    histogramDiMuonPt_fixedBinning_ = bookHistogram1d_fixedBinning("diMuonPt_fixedBinning", name_, 50, 0., 250.);  
    std::vector<double> diMuonMassBinning;
    double diMuonMass = 0.;
    while ( diMuonMass <= 250. ) {
      diMuonMassBinning.push_back(diMuonMass);
      if      ( diMuonMass <  50. ) diMuonMass +=  50.;
      else if ( diMuonMass <  75. ) diMuonMass +=   5.;
      else if ( diMuonMass < 105. ) diMuonMass +=   1.;
      else if ( diMuonMass < 130. ) diMuonMass +=   5.;
      else if ( diMuonMass < 150. ) diMuonMass +=  20.;
      else                          diMuonMass += 100.;
    }
    histogramDiMuonMass_varBinning_ = bookHistogram1d_varBinning("diMuonMass_varBinning", name_, diMuonMassBinning);
    histogramDiMuonMass_fixedBinning_ = bookHistogram1d_fixedBinning("diMuonMass_fixedBinning", name_, 50, 0., 250.);
    histogramDiMuonMass_vs_diMuonPt_varBinning_ = bookHistogram2d_varBinning("diMuonMass_vs_diMuonPt_varBinning", name_, diMuonPtBinning, diMuonMassBinning);
    histogramNumJetsCorrPtGt20_ = bookHistogram1d_fixedBinning("numJetsCorrPtGt20", name_, 20, -0.5, +19.5);
    histogramNumJetsCorrPtGt30_ = bookHistogram1d_fixedBinning("numJetsCorrPtGt30", name_, 20, -0.5, +19.5);
    histogramPFMEt_ = bookHistogram1d_fixedBinning("pfMEt", name_, 50, 0., 250.);
    histogramCaloMEt_ = bookHistogram1d_fixedBinning("caloMEt", name_, 50, 0., 250.);
  }
  ~plotEntryType()
  {
    delete histogramMuon1Pt_varBinning_;
    delete histogramMuon1Pt_fixedBinning_;
    delete histogramMuon1Eta_;
    delete histogramMuon2Pt_varBinning_;
    delete histogramMuon2Pt_fixedBinning_;
    delete histogramMuon2Eta_;
    delete histogramMuon2Pt_vs_muon1Pt_varBinning_;
    delete histogramMuon2Eta_vs_muon1Eta_;
    delete histogramDiMuonPt_varBinning_;
    delete histogramDiMuonPt_fixedBinning_;
    delete histogramDiMuonMass_varBinning_;;
    delete histogramDiMuonMass_fixedBinning_;
    delete histogramDiMuonMass_vs_diMuonPt_varBinning_;
    delete histogramNumJetsCorrPtGt20_;
    delete histogramNumJetsCorrPtGt30_;
    delete histogramPFMEt_;
    delete histogramCaloMEt_;
  }
  TH1* bookHistogram1d_fixedBinning(const TString& name1, const TString& name2, Int_t numBinsX, Float_t xMin, Float_t xMax)
  {
    TString histogramName = Form("%s_%s", name1.Data(), name2.Data());
    TH1* histogram = new TH1D(histogramName.Data(), histogramName.Data(), numBinsX, xMin, xMax);
    return histogram;
  }
  TH1* bookHistogram1d_varBinning(const TString& name1, const TString& name2, const std::vector<double>& binningX)
  {
    TString histogramName = Form("%s_%s", name1.Data(), name2.Data());
    TArrayF binningX_array(binningX.size());
    for ( size_t iBinX = 0; iBinX < binningX.size(); ++iBinX ) {
      binningX_array[iBinX] = binningX[iBinX];
    }
    Int_t numBinsX = binningX_array.GetSize() - 1;
    TH1* histogram = new TH1D(histogramName.Data(), histogramName.Data(), numBinsX, binningX_array.GetArray());
    return histogram;
  }
  TH2* bookHistogram2d_fixedBinning(const TString& name1, const TString& name2, Int_t numBinsX, Float_t xMin, Float_t xMax, Int_t numBinsY, Float_t yMin, Float_t yMax)
  {
    TString histogramName = Form("%s_%s", name1.Data(), name2.Data());
    TH2* histogram = new TH2D(histogramName.Data(), histogramName.Data(), numBinsX, xMin, xMax, numBinsY, yMin, yMax);
    return histogram;
  }
  TH2* bookHistogram2d_varBinning(const TString& name1, const TString& name2, const std::vector<double>& binningX, const std::vector<double>& binningY)
  {
    TString histogramName = Form("%s_%s", name1.Data(), name2.Data());
    TArrayF binningX_array(binningX.size());
    for ( size_t iBinX = 0; iBinX < binningX.size(); ++iBinX ) {
      binningX_array[iBinX] = binningX[iBinX];
    }
    Int_t numBinsX = binningX_array.GetSize() - 1;
    TArrayF binningY_array(binningY.size());
    for ( size_t iBinY = 0; iBinY < binningY.size(); ++iBinY ) {
      binningY_array[iBinY] = binningY[iBinY];
    }
    Int_t numBinsY = binningY_array.GetSize() - 1;
    TH2* histogram = new TH2D(histogramName.Data(), histogramName.Data(), numBinsX, binningX_array.GetArray(), numBinsY, binningY_array.GetArray());
    return histogram;
  }
  void fillHistograms(TTree* tree, const std::vector<weightEntryType*>& weightEntries, int maxEvents)
  {
    Float_t diMuonEn, diMuonPx, diMuonPy, diMuonPz;
    Float_t muon1En, muon1Px, muon1Py, muon1Pz;
    Float_t muon2En, muon2Px, muon2Py, muon2Pz;
    Int_t isValid;
    Int_t numJetsCorrPtGt20, numJetsCorrPtGt30;
    Float_t pfMEt, caloMEt;

    Float_t muonRadCorrWeight;
    Float_t TauSpinnerWeight;
    Float_t genFilterWeight;
    Float_t embeddingKineReweight1;
    Float_t embeddingKineReweight2;
    Float_t embeddingKineReweight3;
    Float_t embeddingKineReweight4;
    Float_t evtSelEffCorrWeight;
    Float_t pileUpReweight;

    if ( type_ == kZtautau || type_ == kGenEmbedded || type_ == kRecEmbedded ) {
      tree->SetBranchAddress("genDiTauEn", &diMuonEn);
      tree->SetBranchAddress("genDiTauPx", &diMuonPx);
      tree->SetBranchAddress("genDiTauPy", &diMuonPy);
      tree->SetBranchAddress("genDiTauPz", &diMuonPz);
      tree->SetBranchAddress("genTau1En", &muon1En);
      tree->SetBranchAddress("genTau1Px", &muon1Px);
      tree->SetBranchAddress("genTau1Py", &muon1Py);
      tree->SetBranchAddress("genTau1Pz", &muon1Pz);
      tree->SetBranchAddress("genTau2En", &muon2En);
      tree->SetBranchAddress("genTau2Px", &muon2Px);
      tree->SetBranchAddress("genTau2Py", &muon2Py);
      tree->SetBranchAddress("genTau2Pz", &muon2Pz);
      isValid = true;
    } else if ( type_ == kGenZmumu ) {
      tree->SetBranchAddress("genDiMuonEn", &diMuonEn);
      tree->SetBranchAddress("genDiMuonPx", &diMuonPx);
      tree->SetBranchAddress("genDiMuonPy", &diMuonPy);
      tree->SetBranchAddress("genDiMuonPz", &diMuonPz);
      tree->SetBranchAddress("genMuonPlusEn", &muon1En);
      tree->SetBranchAddress("genMuonPlusPx", &muon1Px);
      tree->SetBranchAddress("genMuonPlusPy", &muon1Py);
      tree->SetBranchAddress("genMuonPlusPz", &muon1Pz);
      tree->SetBranchAddress("genMuonMinusEn", &muon2En);
      tree->SetBranchAddress("genMuonMinusPx", &muon2Px);
      tree->SetBranchAddress("genMuonMinusPy", &muon2Py);
      tree->SetBranchAddress("genMuonMinusPz", &muon2Pz);
      tree->SetBranchAddress("genDiMuonIsValid", &isValid);
    } else if ( type_ == kRecZmumu ) {
      tree->SetBranchAddress("recDiMuonEn", &diMuonEn);
      tree->SetBranchAddress("recDiMuonPx", &diMuonPx);
      tree->SetBranchAddress("recDiMuonPy", &diMuonPy);
      tree->SetBranchAddress("recDiMuonPz", &diMuonPz);
      tree->SetBranchAddress("recMuonPlusEn", &muon1En);
      tree->SetBranchAddress("recMuonPlusPx", &muon1Px);
      tree->SetBranchAddress("recMuonPlusPy", &muon1Py);
      tree->SetBranchAddress("recMuonPlusPz", &muon1Pz);
      tree->SetBranchAddress("recMuonMinusEn", &muon2En);
      tree->SetBranchAddress("recMuonMinusPx", &muon2Px);
      tree->SetBranchAddress("recMuonMinusPy", &muon2Py);
      tree->SetBranchAddress("recMuonMinusPz", &muon2Pz);
      tree->SetBranchAddress("recDiMuonIsValid", &isValid);
    } else assert(0);
    tree->SetBranchAddress("numJetsCorrPtGt20", &numJetsCorrPtGt20);
    tree->SetBranchAddress("numJetsCorrPtGt30", &numJetsCorrPtGt30);
    tree->SetBranchAddress("recPFMEtPt", &pfMEt);
    tree->SetBranchAddress("recCaloMEtPt", &caloMEt);

    if ( type_ == kGenEmbedded || type_ == kRecEmbedded ) {
      //tree->SetBranchAddress("muonRadiationCorrWeightProducer_weight", &muonRadCorrWeight);
      tree->SetBranchAddress("TauSpinnerReco_TauSpinnerWT", &TauSpinnerWeight);
      tree->SetBranchAddress("genFilterInfo", &genFilterWeight);
      //-------------------------------------------------------------------------
      // CV: to be enabled for DEBUGging purposes only !!
      //tree->SetBranchAddress("embeddingKineReweightGENtoEmbedded_genDiTauMassVsGenDiTauPt", &embeddingKineReweight1);
      //tree->SetBranchAddress("embeddingKineReweightGENtoEmbedded_genTau2PtVsGenTau1Pt", &embeddingKineReweight2);
      //tree->SetBranchAddress("embeddingKineReweightGENtoREC_genDiTauMassVsGenDiTauPt", &embeddingKineReweight3);
      //tree->SetBranchAddress("embeddingKineReweightGENtoREC_genTau2PtVsGenTau1Pt", &embeddingKineReweight4);
      //-------------------------------------------------------------------------
    }
    if ( type_ == kRecEmbedded ) {
      tree->SetBranchAddress("ZmumuEvtSelEffCorrWeightProducer_weight", &evtSelEffCorrWeight);
    }
    tree->SetBranchAddress("vertexMultiplicityReweight2012RunABCDruns190456to208686_", &pileUpReweight);

    int numEntries = tree->GetEntries();
    for ( int iEntry = 0; iEntry < numEntries && (iEntry < maxEvents || maxEvents == -1); ++iEntry ) {
      if ( iEntry > 0 && (iEntry % 10000) == 0 ) {
	std::cout << "processing Event " << iEntry << std::endl;
      }
      
      tree->GetEntry(iEntry);

      if ( !isValid ) continue;

      TLorentzVector diMuonP4(diMuonPx, diMuonPy, diMuonPz, diMuonEn);
      TLorentzVector muon1P4(muon1Px, muon1Py, muon1Pz, muon1En);
      TLorentzVector muon2P4(muon2Px, muon2Py, muon2Pz, muon2En);

      Float_t evtWeight = 1.0;
      if ( type_ == kGenEmbedded || type_ == kRecEmbedded ) {
	//evtWeight *= muonRadCorrWeight;
	evtWeight *= TauSpinnerWeight;
	evtWeight *= genFilterWeight;
	//-------------------------------------------------------------------------
	// CV: to be enabled for DEBUGging purposes only !!
	//evtWeight *= embeddingKineReweight1;
	//evtWeight *= embeddingKineReweight2;
	//evtWeight *= embeddingKineReweight3;
	//evtWeight *= embeddingKineReweight4;
	//-------------------------------------------------------------------------
      }
      if ( type_ == kRecEmbedded ) {
	evtWeight *= evtSelEffCorrWeight;
      }
      evtWeight *= pileUpReweight;
      for ( std::vector<weightEntryType*>::const_iterator weightEntry = weightEntries.begin();
	    weightEntry != weightEntries.end(); ++weightEntry ) {
	//std::cout << (*weightEntry)->name_ << " = " << (**weightEntry)(muon1P4.Pt(), muon1P4.Eta(), muon2P4.Pt(), muon2P4.Eta(), diMuonP4.Pt(), diMuonP4.M()) << std::endl;
	evtWeight *= (**weightEntry)(muon1P4.Pt(), muon1P4.Eta(), muon2P4.Pt(), muon2P4.Eta(), diMuonP4.Pt(), diMuonP4.M());	
      }

      if ( evtWeight < 1.e-3 || evtWeight > 1.e+3 || TMath::IsNaN(evtWeight) ) {  
	//if ( type_ == kGenEmbedded || type_ == kRecEmbedded ) {
	//  std::cout << "muonRadCorrWeight = " << muonRadCorrWeight << std::endl;
	//  std::cout << "TauSpinnerWeight = " << TauSpinnerWeight << std::endl;
	//  std::cout << "genFilterWeight = " << genFilterWeight << std::endl;
	//}
	//if ( type_ == kRecEmbedded ) {
	//  std::cout << "evtSelEffCorrWeight = " << evtSelEffCorrWeight << std::endl;
	//}
	//for ( std::vector<weightEntryType*>::const_iterator weightEntry = weightEntries.begin();
	//      weightEntry != weightEntries.end(); ++weightEntry ) {
	//  std::cout << (*weightEntry)->name_ << " = " << (**weightEntry)(muon1P4.Pt(), muon1P4.Eta(), muon2P4.Pt(), muon2P4.Eta(), diMuonP4.Pt(), diMuonP4.M()) << std::endl;
	//}
	continue;
      }

      histogramMuon1Pt_varBinning_->Fill(muon1P4.Pt(), evtWeight);
      histogramMuon1Pt_fixedBinning_->Fill(muon1P4.Pt(), evtWeight);
      histogramMuon1Eta_->Fill(muon1P4.Eta(), evtWeight);
      histogramMuon2Pt_varBinning_->Fill(muon2P4.Pt(), evtWeight);
      histogramMuon2Pt_fixedBinning_->Fill(muon2P4.Pt(), evtWeight);
      histogramMuon2Eta_->Fill(muon2P4.Eta(), evtWeight);
      histogramMuon2Pt_vs_muon1Pt_varBinning_->Fill(muon1P4.Pt(), muon2P4.Pt(), evtWeight);
      histogramMuon2Eta_vs_muon1Eta_->Fill(muon1P4.Eta(), muon2P4.Eta(), evtWeight);
      histogramDiMuonPt_varBinning_->Fill(diMuonP4.Pt(), evtWeight);
      histogramDiMuonPt_fixedBinning_->Fill(diMuonP4.Pt(), evtWeight);
      histogramDiMuonMass_varBinning_->Fill(diMuonP4.M(), evtWeight);
      histogramDiMuonMass_fixedBinning_->Fill(diMuonP4.M(), evtWeight);      
      histogramDiMuonMass_vs_diMuonPt_varBinning_->Fill(diMuonP4.Pt(), diMuonP4.M(), evtWeight);
      histogramNumJetsCorrPtGt20_->Fill(numJetsCorrPtGt20, evtWeight);
      histogramNumJetsCorrPtGt30_->Fill(numJetsCorrPtGt30, evtWeight);
      histogramPFMEt_->Fill(pfMEt, evtWeight);
      histogramCaloMEt_->Fill(caloMEt, evtWeight);
    }
  }
  void saveHistograms()
  {
    histogramMuon1Pt_varBinning_->Write();
    histogramMuon1Pt_fixedBinning_->Write();
    histogramMuon1Eta_->Write();
    histogramMuon2Pt_varBinning_->Write();
    histogramMuon2Pt_fixedBinning_->Write();
    histogramMuon2Eta_->Write();
    histogramMuon2Pt_vs_muon1Pt_varBinning_->Write();
    histogramMuon2Eta_vs_muon1Eta_->Write();
    histogramDiMuonPt_varBinning_->Write();
    histogramDiMuonPt_fixedBinning_->Write();
    histogramDiMuonMass_varBinning_->Write();
    histogramDiMuonMass_fixedBinning_->Write();
    histogramDiMuonMass_vs_diMuonPt_varBinning_->Write();
    histogramNumJetsCorrPtGt20_->Write();
    histogramNumJetsCorrPtGt30_->Write();
    histogramPFMEt_->Write();
    histogramCaloMEt_->Write();
  }
  enum { kZtautau, kGenEmbedded, kRecEmbedded, kGenZmumu, kRecZmumu };
  int type_;
  std::string name_;
  TH1* histogramMuon1Pt_varBinning_;
  TH1* histogramMuon1Pt_fixedBinning_;
  TH1* histogramMuon1Eta_;
  TH1* histogramMuon2Pt_varBinning_;
  TH1* histogramMuon2Pt_fixedBinning_;
  TH1* histogramMuon2Eta_;
  TH2* histogramMuon2Pt_vs_muon1Pt_varBinning_;
  TH2* histogramMuon2Eta_vs_muon1Eta_;
  TH1* histogramDiMuonPt_varBinning_;
  TH1* histogramDiMuonPt_fixedBinning_;
  TH1* histogramDiMuonMass_varBinning_;
  TH1* histogramDiMuonMass_fixedBinning_;
  TH2* histogramDiMuonMass_vs_diMuonPt_varBinning_;
  TH1* histogramNumJetsCorrPtGt20_;
  TH1* histogramNumJetsCorrPtGt30_;
  TH1* histogramPFMEt_;
  TH1* histogramCaloMEt_;
};

//-------------------------------------------------------------------------------
TH1* compRatioHistogram(const std::string& ratioHistogramName, const TH1* numerator, const TH1* denominator, Float_t offset = 0., bool normalize = true)
{
  //std::cout << "<compRatioHistogram>:" << std::endl;
  //std::cout << " numerator: name = " << numerator->GetName() << ", integral = " << numerator->Integral() << std::endl;
  //std::cout << " denominator: name = " << denominator->GetName() << ", integral = " << denominator->Integral() << std::endl;

  assert(numerator->GetDimension() == denominator->GetDimension());
  assert(numerator->GetNbinsX() == denominator->GetNbinsX());
  assert(numerator->GetNbinsY() == denominator->GetNbinsY());

  TH1* histogramRatio = (TH1*)numerator->Clone(ratioHistogramName.data());
  histogramRatio->Divide(denominator);
  if ( normalize ) histogramRatio->Scale(denominator->Integral()/numerator->Integral());
  
  if ( offset != 0. ) {    
    if ( numerator->GetDimension() == 1 ) {
      int numBinsX = histogramRatio->GetNbinsX();
      for ( int iBinX = 1; iBinX <= numBinsX; ++iBinX ){
	double binContent = histogramRatio->GetBinContent(iBinX);
	histogramRatio->SetBinContent(iBinX, binContent - offset);
      }
    } else if ( numerator->GetDimension() == 2 ) {
      int numBinsX = histogramRatio->GetNbinsX();
      int numBinsY = histogramRatio->GetNbinsY();
      for ( int iBinX = 1; iBinX <= numBinsX; ++iBinX ){
	for ( int iBinY = 1; iBinY <= numBinsY; ++iBinY ){
	  double binContent = histogramRatio->GetBinContent(iBinX, iBinY);
	  histogramRatio->SetBinContent(iBinX, iBinY, binContent - offset);
	}
      }
    } else assert(0);
  }

  histogramRatio->SetLineColor(numerator->GetLineColor());
  histogramRatio->SetLineWidth(numerator->GetLineWidth());
  histogramRatio->SetMarkerColor(numerator->GetMarkerColor());
  histogramRatio->SetMarkerStyle(numerator->GetMarkerStyle());

  return histogramRatio;
}

double square(double x)
{
  return x*x;
}

TH1* rebinHistogram(const TH1* histogram, unsigned numBinsMin_rebinned, double xMin, double xMax, bool normalize)
{
  TH1* histogram_rebinned = 0;

  if ( histogram ) {
    unsigned numBins = histogram->GetNbinsX();
    unsigned numBins_withinRange = 0;
    for ( unsigned iBin = 1; iBin <= numBins; ++iBin ) {
      double binCenter = histogram->GetBinCenter(iBin);
      if ( binCenter >= xMin && binCenter <= xMax ) ++numBins_withinRange;
    }

    std::cout << "histogram = " << histogram->GetName() << ":" 
              << " numBins(" << xMin << ".." << "xMax) = " << numBins_withinRange << ", integral = " << histogram->Integral() << std::endl;
    
    unsigned numBins_rebinned = numBins_withinRange;

    for ( int combineNumBins = 5; combineNumBins >= 2; --combineNumBins ) {
      if ( numBins_withinRange >= (combineNumBins*numBinsMin_rebinned) && (numBins % combineNumBins) == 0 ) {
        numBins_rebinned /= combineNumBins;
        numBins_withinRange /= combineNumBins;
      }
    }

    std::string histogramName_rebinned = std::string(histogram->GetName()).append("_rebinned");
    histogram_rebinned = new TH1D(histogramName_rebinned.data(), histogram->GetTitle(), numBins_rebinned, xMin, xMax);
    if ( !histogram_rebinned->GetSumw2N() ) histogram_rebinned->Sumw2();

    TAxis* xAxis = histogram_rebinned->GetXaxis();
      
    unsigned iBin = 1;
    for ( unsigned iBin_rebinned = 1; iBin_rebinned <= numBins_rebinned; ++iBin_rebinned ) {
      double binContent_rebinned = 0.;
      double binError2_rebinned = 0.;

      double xMin_rebinnedBin = xAxis->GetBinLowEdge(iBin_rebinned);
      double xMax_rebinnedBin = xAxis->GetBinUpEdge(iBin_rebinned);

      while ( histogram->GetBinCenter(iBin) < xMin_rebinnedBin ) {
	++iBin;
      }

      while ( histogram->GetBinCenter(iBin) >= xMin_rebinnedBin && histogram->GetBinCenter(iBin) < xMax_rebinnedBin ) {
	binContent_rebinned += histogram->GetBinContent(iBin);
	binError2_rebinned += square(histogram->GetBinError(iBin));
	++iBin;
      }

      histogram_rebinned->SetBinContent(iBin_rebinned, binContent_rebinned);
      histogram_rebinned->SetBinError(iBin_rebinned, TMath::Sqrt(binError2_rebinned));
    }

    if ( normalize ) {
      if ( !histogram_rebinned->GetSumw2N() ) histogram_rebinned->Sumw2();
      histogram_rebinned->Scale(1./histogram_rebinned->Integral());
    }

    std::cout << "histogram(rebinned) = " << histogram_rebinned->GetName() << ":" 
              << " numBins = " << histogram_rebinned->GetNbinsX() << ", integral = " << histogram_rebinned->Integral() << std::endl;
  }

  return histogram_rebinned;
}

void showDistribution(double canvasSizeX, double canvasSizeY,
		      TH1* histogram_ref, const std::string& legendEntry_ref,
		      TH1* histogram2, const std::string& legendEntry2,
		      TH1* histogram3, const std::string& legendEntry3,
		      TH1* histogram4, const std::string& legendEntry4,
		      TH1* histogram5, const std::string& legendEntry5,
		      TH1* histogram6, const std::string& legendEntry6,
		      double xMin, double xMax, unsigned numBinsMin_rebinned, const std::string& xAxisTitle, double xAxisOffset,
		      bool useLogScale, double yMin, double yMax, const std::string& yAxisTitle, double yAxisOffset,
		      double legendX0, double legendY0, 
		      const std::string& outputFileName)
{
  TCanvas* canvas = new TCanvas("canvas", "canvas", canvasSizeX, canvasSizeY);
  canvas->SetFillColor(10);
  canvas->SetBorderSize(2);
  canvas->SetLeftMargin(0.12);
  canvas->SetBottomMargin(0.12);

  TPad* topPad = new TPad("topPad", "topPad", 0.00, 0.35, 1.00, 1.00);
  topPad->SetFillColor(10);
  topPad->SetTopMargin(0.04);
  topPad->SetLeftMargin(0.15);
  topPad->SetBottomMargin(0.03);
  topPad->SetRightMargin(0.05);
  topPad->SetLogy(useLogScale);

  TPad* bottomPad = new TPad("bottomPad", "bottomPad", 0.00, 0.00, 1.00, 0.35);
  bottomPad->SetFillColor(10);
  bottomPad->SetTopMargin(0.02);
  bottomPad->SetLeftMargin(0.15);
  bottomPad->SetBottomMargin(0.24);
  bottomPad->SetRightMargin(0.05);
  bottomPad->SetLogy(false);

  canvas->cd();
  topPad->Draw();
  topPad->cd();

  int colors[6] = { 1, 2, 3, 4, 6, 7 };
  int markerStyles[6] = { 22, 32, 20, 24, 21, 25 };

  TLegend* legend = new TLegend(legendX0, legendY0, 0.94, 0.94, "", "brNDC"); 
  legend->SetBorderSize(0);
  legend->SetFillColor(0);

  TH1* histogram_ref_rebinned = rebinHistogram(histogram_ref, numBinsMin_rebinned, xMin, xMax, true);
  histogram_ref_rebinned->SetTitle("");
  histogram_ref_rebinned->SetStats(false);
  histogram_ref_rebinned->SetMinimum(yMin);
  histogram_ref_rebinned->SetMaximum(yMax);
  histogram_ref_rebinned->SetLineColor(colors[0]);
  histogram_ref_rebinned->SetLineWidth(2);
  histogram_ref_rebinned->SetMarkerColor(colors[0]);
  histogram_ref_rebinned->SetMarkerStyle(markerStyles[0]);
  histogram_ref_rebinned->Draw("e1p");
  legend->AddEntry(histogram_ref_rebinned, legendEntry_ref.data(), "p");

  TAxis* xAxis_top = histogram_ref_rebinned->GetXaxis();
  xAxis_top->SetTitle(xAxisTitle.data());
  xAxis_top->SetTitleOffset(xAxisOffset);
  xAxis_top->SetLabelColor(10);
  xAxis_top->SetTitleColor(10);

  TAxis* yAxis_top = histogram_ref_rebinned->GetYaxis();
  yAxis_top->SetTitle(yAxisTitle.data());
  yAxis_top->SetTitleOffset(yAxisOffset);

  TH1* histogram2_rebinned = 0;
  if ( histogram2 ) {
    histogram2_rebinned = rebinHistogram(histogram2, numBinsMin_rebinned, xMin, xMax, true);
    histogram2_rebinned->SetLineColor(colors[1]);
    histogram2_rebinned->SetLineWidth(2);
    histogram2_rebinned->SetMarkerColor(colors[1]);
    histogram2_rebinned->SetMarkerStyle(markerStyles[1]);
    histogram2_rebinned->Draw("e1psame");
    legend->AddEntry(histogram2_rebinned, legendEntry2.data(), "p");
  }

  TH1* histogram3_rebinned = 0;
  if ( histogram3 ) {
    histogram3_rebinned = rebinHistogram(histogram3, numBinsMin_rebinned, xMin, xMax, true);
    histogram3_rebinned->SetLineColor(colors[2]);
    histogram3_rebinned->SetLineWidth(2);
    histogram3_rebinned->SetMarkerColor(colors[2]);
    histogram3_rebinned->SetMarkerStyle(markerStyles[2]);
    histogram3_rebinned->Draw("e1psame");
    legend->AddEntry(histogram3_rebinned, legendEntry3.data(), "p");
  }

  TH1* histogram4_rebinned = 0;
  if ( histogram4 ) {
    histogram4_rebinned = rebinHistogram(histogram4, numBinsMin_rebinned, xMin, xMax, true);
    histogram4_rebinned->SetLineColor(colors[3]);
    histogram4_rebinned->SetLineWidth(2);
    histogram4_rebinned->SetMarkerColor(colors[3]);
    histogram4_rebinned->SetMarkerStyle(markerStyles[3]);
    histogram4_rebinned->Draw("e1psame");
    legend->AddEntry(histogram4_rebinned, legendEntry4.data(), "p");
  }

  TH1* histogram5_rebinned = 0;
  if ( histogram5 ) {
    histogram5_rebinned = rebinHistogram(histogram5, numBinsMin_rebinned, xMin, xMax, true);
    histogram5_rebinned->SetLineColor(colors[4]);
    histogram5_rebinned->SetLineWidth(2);
    histogram5_rebinned->SetMarkerColor(colors[4]);
    histogram5_rebinned->SetMarkerStyle(markerStyles[4]);
    histogram5_rebinned->Draw("e1psame");
    legend->AddEntry(histogram5_rebinned, legendEntry5.data(), "p");
  }

  TH1* histogram6_rebinned = 0;
  if ( histogram6 ) {
    histogram6_rebinned = rebinHistogram(histogram6, numBinsMin_rebinned, xMin, xMax, true);
    histogram6_rebinned->SetLineColor(colors[5]);
    histogram6_rebinned->SetLineWidth(2);
    histogram6_rebinned->SetMarkerColor(colors[5]);
    histogram6_rebinned->SetMarkerStyle(markerStyles[5]);
    histogram6_rebinned->Draw("e1psame");
    legend->AddEntry(histogram6_rebinned, legendEntry6.data(), "p");
  }

  legend->Draw();

  canvas->cd();
  bottomPad->Draw();
  bottomPad->cd();

  TH1* histogram2_div_ref = 0;
  if ( histogram2 ) {
    std::string histogramName2_div_ref = std::string(histogram2->GetName()).append("_div_").append(histogram_ref->GetName());
    histogram2_div_ref = compRatioHistogram(histogramName2_div_ref, histogram2_rebinned, histogram_ref_rebinned, 1.0);
    histogram2_div_ref->SetTitle("");
    histogram2_div_ref->SetStats(false);
    histogram2_div_ref->SetMinimum(-0.50);
    histogram2_div_ref->SetMaximum(+0.50);

    TAxis* xAxis_bottom = histogram2_div_ref->GetXaxis();
    xAxis_bottom->SetTitle(xAxis_top->GetTitle());
    xAxis_bottom->SetLabelColor(1);
    xAxis_bottom->SetTitleColor(1);
    xAxis_bottom->SetTitleOffset(1.20);
    xAxis_bottom->SetTitleSize(0.08);
    xAxis_bottom->SetLabelOffset(0.02);
    xAxis_bottom->SetLabelSize(0.08);
    xAxis_bottom->SetTickLength(0.055);
    
    TAxis* yAxis_bottom = histogram2_div_ref->GetYaxis();
    yAxis_bottom->SetTitle("#frac{Embedding - Z/#gamma^{*} #rightarrow #muon #muon}{Z/#gamma^{*} #rightarrow #muon #muon}");
    yAxis_bottom->SetTitleOffset(0.70);
    yAxis_bottom->SetNdivisions(505);
    yAxis_bottom->CenterTitle();
    yAxis_bottom->SetTitleSize(0.08);
    yAxis_bottom->SetLabelSize(0.08);
    yAxis_bottom->SetTickLength(0.04);  
  
    histogram2_div_ref->Draw("e1p");
  }

  TH1* histogram3_div_ref = 0;
  if ( histogram3 ) {
    std::string histogramName3_div_ref = std::string(histogram3->GetName()).append("_div_").append(histogram_ref->GetName());
    histogram3_div_ref = compRatioHistogram(histogramName3_div_ref, histogram3_rebinned, histogram_ref_rebinned, 1.0);
    histogram3_div_ref->Draw("e1psame");
  }

  TH1* histogram4_div_ref = 0;
  if ( histogram4 ) {
    std::string histogramName4_div_ref = std::string(histogram4->GetName()).append("_div_").append(histogram_ref->GetName());
    histogram4_div_ref = compRatioHistogram(histogramName4_div_ref, histogram4_rebinned, histogram_ref_rebinned, 1.0);
    histogram4_div_ref->Draw("e1psame");
  }

  TH1* histogram5_div_ref = 0;
  if ( histogram5 ) {
    std::string histogramName5_div_ref = std::string(histogram5->GetName()).append("_div_").append(histogram_ref->GetName());
    histogram5_div_ref = compRatioHistogram(histogramName5_div_ref, histogram5_rebinned, histogram_ref_rebinned, 1.0);
    histogram5_div_ref->Draw("e1psame");
  }

  TH1* histogram6_div_ref = 0;
  if ( histogram6 ) {
    std::string histogramName6_div_ref = std::string(histogram6->GetName()).append("_div_").append(histogram_ref->GetName());
    histogram6_div_ref = compRatioHistogram(histogramName6_div_ref, histogram6_rebinned, histogram_ref_rebinned, 1.0);
    histogram6_div_ref->Draw("e1psame");
  }
  
  canvas->Update();
  size_t idx = outputFileName.find_last_of('.');
  std::string outputFileName_plot = std::string(outputFileName, 0, idx);
  if ( useLogScale ) outputFileName_plot.append("_log");
  else outputFileName_plot.append("_linear");
  if ( idx != std::string::npos ) canvas->Print(std::string(outputFileName_plot).append(std::string(outputFileName, idx)).data());
  canvas->Print(std::string(outputFileName_plot).append(".png").data());
  //canvas->Print(std::string(outputFileName_plot).append(".pdf").data());
  
  delete legend;
  delete histogram2_div_ref;
  delete histogram3_div_ref;
  delete histogram4_div_ref;
  delete histogram5_div_ref;
  delete histogram6_div_ref;
  delete topPad;
  delete bottomPad;
  delete canvas;  
}
//-------------------------------------------------------------------------------

void makeEmbeddingKineReweightLUTs2()
{
  gROOT->SetBatch(true);

  TString inputFilePath = "/data1/veelken/tmp/EmbeddingValidation/";

  //std::string channel = "etau";
  //std::string channel = "etau_soft";
  //std::string channel = "mutau";
  std::string channel = "mutauNegAngle";
  //std::string channel = "mutau_soft";
  //std::string channel = "emu";
  //std::string channel = "tautau";

  std::string mode = "recEmbedded";
  //std::string mode = "genEmbedded";
  //std::string mode = "Zmumu";

  TString inputFileName_Ztautau;
  TString inputFileName_genEmbedded;
  TString inputFileName_recEmbedded;
  TString inputFileName_Zmumu = "embeddingKineReweightNtuple_all_2013Mar03.root";
  if ( channel == "etau" ) {
    inputFileName_Ztautau  = "embeddingKineReweightNtuple_simDYtoTauTau_etau_all_v2_4_0.root";
    inputFileName_genEmbedded = "embeddingKineReweightNtuple_simDYtoMuMu_embedEqRH_cleanEqDEDX_replaceGenMuons_by_ePtGt20tauPtGt18_embedAngleEq90_noPolarization_wTauSpinner_all_v2_4_0.root";  
    inputFileName_recEmbedded = "embeddingKineReweightNtuple_simDYtoMuMu_embedEqRH_cleanEqDEDX_replaceRecMuons_by_ePtGt20tauPtGt18_embedAngleEq90_noPolarization_wTauSpinner_all_v2_4_0.root";
  } else if ( channel == "etau_soft" ) {
    inputFileName_Ztautau  = "embeddingKineReweightNtuple_simDYtoTauTau_etau_soft_all_v2_4_0.root";
    inputFileName_genEmbedded = "embeddingKineReweightNtuple_simDYtoMuMu_embedEqRH_cleanEqDEDX_replaceGenMuons_by_ePt9to30tauPtGt15_embedAngleEq90_noPolarization_wTauSpinner_all_v2_4_0.root";  
    inputFileName_recEmbedded = "embeddingKineReweightNtuple_simDYtoMuMu_embedEqRH_cleanEqDEDX_replaceRecMuons_by_ePt9to30tauPtGt15_embedAngleEq90_noPolarization_wTauSpinner_all_v2_4_0.root"; 
  } else if ( channel == "mutau" ) {
    inputFileName_Ztautau  = "embeddingKineReweightNtuple_simDYtoTauTau_mutau_all_v2_4_0.root";
    inputFileName_genEmbedded = "embeddingKineReweightNtuple_simDYtoMuMu_embedEqRH_cleanEqDEDX_replaceGenMuons_by_muPtGt16tauPtGt18_embedAngleEq90_noPolarization_wTauSpinner_all_v2_4_0.root";
    inputFileName_recEmbedded = "embeddingKineReweightNtuple_simDYtoMuMu_embedEqRH_cleanEqDEDX_replaceRecMuons_by_muPtGt16tauPtGt18_embedAngleEq90_noPolarization_wTauSpinner_all_v2_4_0.root";
  } else if ( channel == "mutauNegAngle" ) {
    inputFileName_Ztautau  = "embeddingKineReweightNtuple_simDYtoTauTau_mutau_all_v2_6_4.root";
    inputFileName_genEmbedded = "embeddingKineReweightNtuple_simDYtoMuMu_embedEqRH_cleanEqDEDX_replaceGenMuons_by_muPtGt16tauPtGt18_embedAngleEqMinus90_noPolarization_wTauSpinner_all_v2_6_4.root";
    inputFileName_recEmbedded = "embeddingKineReweightNtuple_simDYtoMuMu_embedEqRH_cleanEqDEDX_replaceRecMuons_by_muPtGt16tauPtGt18_embedAngleEqMinus90_noPolarization_wTauSpinner_all_v2_6_4.root";
  } else if ( channel == "mutau_soft" ) {
    inputFileName_Ztautau  = "embeddingKineReweightNtuple_simDYtoTauTau_mutau_soft_all_v2_4_0.root";
    inputFileName_genEmbedded = "embeddingKineReweightNtuple_simDYtoMuMu_embedEqRH_cleanEqDEDX_replaceGenMuons_by_muPt7to25tauPtGt15_embedAngleEq90_noPolarization_wTauSpinner_all_v2_4_0.root";
    inputFileName_recEmbedded = "embeddingKineReweightNtuple_simDYtoMuMu_embedEqRH_cleanEqDEDX_replaceRecMuons_by_muPt7to25tauPtGt15_embedAngleEq90_noPolarization_wTauSpinner_all_v2_4_0.root";
  } else if ( channel == "emu" ) {
    inputFileName_Ztautau  = "embeddingKineReweightNtuple_simDYtoTauTau_emu_all_v2_1_6_kineReweighted.root";
    inputFileName_genEmbedded = "embeddingKineReweightNtuple_simDYtoMuMu_embedEqRH_cleanEqDEDX_replaceGenMuons_by_emu_embedAngleEq90_noPolarization_wTauSpinner_all_v2_1_6_kineReweighted.root";  
    inputFileName_recEmbedded = "embeddingKineReweightNtuple_simDYtoMuMu_embedEqRH_cleanEqDEDX_replaceRecMuons_by_emu_embedAngleEq90_noPolarization_wTauSpinner_all_v2_1_6_kineReweighted.root";
  } else if ( channel == "tautau" ) {
    inputFileName_Ztautau  = "embeddingKineReweightNtuple_simDYtoTauTau_tautau_all_v2_2_0.root";
    inputFileName_genEmbedded = "embeddingKineReweightNtuple_simDYtoMuMu_embedEqRH_cleanEqDEDX_replaceGenMuons_by_tauPtGt30tauPtGt30_embedAngleEq90_noPolarization_wTauSpinner_all_v2_2_0.root";  
    inputFileName_recEmbedded = "embeddingKineReweightNtuple_simDYtoMuMu_embedEqRH_cleanEqDEDX_replaceRecMuons_by_tauPtGt30tauPtGt30_embedAngleEq90_noPolarization_wTauSpinner_all_v2_2_0.root";
  } else {
    std::cout << "Invalid channel = " << channel << " !!" << std::endl;
    assert(0);
  }

  TString inputFileName_Embedded;
  if ( mode == "genEmbedded" ) {
    inputFileName_Embedded = inputFileName_genEmbedded;
  } else if ( mode == "recEmbedded" ) {
    inputFileName_Embedded = inputFileName_recEmbedded;
  } else if ( mode != "Zmumu" ) {
    std::cout << "Invalid mode = " << mode << " !!" << std::endl;
    assert(0);
  }
  
  TString treeName = "embeddingKineReweightNtupleProducerAfterGenAndRecCuts/embeddingKineReweightNtuple";
  //TString treeName = "embeddingKineReweightNtupleProducer/embeddingKineReweightNtuple";

  int maxEvents = -1;

  TFile* inputFile_reference = 0;
  TTree* tree_reference      = 0;
  TFile* inputFile_test      = 0;
  TTree* tree_test           = 0;

  if ( mode == "genEmbedded" || mode == "recEmbedded" ) {
    TFile* inputFile_Ztautau = TFile::Open(TString(inputFilePath).Append(inputFileName_Ztautau).Data());
    if ( !inputFile_Ztautau ) {
      std::cerr << "Failed to open input file = " << inputFileName_Ztautau.Data() << " !!" << std::endl;
      assert(0);
    }
    tree_reference = dynamic_cast<TTree*>(inputFile_Ztautau->Get(treeName.Data()));
    if ( !tree_reference ) {
      std::cerr << "Failed to find tree = " << treeName.Data() << " in input file = " << inputFileName_Ztautau.Data() << " !!" << std::endl;
      assert(0);
    }
    std::cout << "tree_Ztautau has " << tree_reference->GetEntries() << " entries." << std::endl;

    TFile* inputFile_Embedded = TFile::Open(TString(inputFilePath).Append(inputFileName_Embedded).Data());
    if ( !inputFile_Embedded ) {
      std::cerr << "Failed to open input file = " << inputFileName_Embedded.Data() << " !!" << std::endl;
      assert(0);
    }
    tree_test = dynamic_cast<TTree*>(inputFile_Embedded->Get(treeName.Data()));
    if ( !tree_test ) {
      std::cerr << "Failed to find tree = " << treeName.Data() << " in input file = " << inputFileName_Embedded.Data() << " !!" << std::endl;
      assert(0);
    }
    std::cout << "tree_Embedded has " << tree_test->GetEntries() << " entries." << std::endl;
  } else if ( mode == "Zmumu" ) {
    TFile* inputFile_Zmumu = TFile::Open(TString(inputFilePath).Append(inputFileName_Zmumu).Data());
    if ( !inputFile_Zmumu ) {
      std::cerr << "Failed to open input file = " << inputFileName_Zmumu.Data() << " !!" << std::endl;
      assert(0);
    }
    tree_reference = dynamic_cast<TTree*>(inputFile_Zmumu->Get(treeName.Data()));
    if ( !tree_reference ) {
      std::cerr << "Failed to find tree = " << treeName.Data() << " in input file = " << inputFileName_Zmumu.Data() << " !!" << std::endl;
      assert(0);
    }
    std::cout << "tree_Zmumu has " << tree_reference->GetEntries() << " entries." << std::endl;
    tree_test = tree_reference;
  } else {
    std::cerr << "Invalid Configuration Parameter 'mode' = " << mode << " !!" << std::endl;
    assert(0);
  }

  plotEntryType* plots_reference        = 0;
  plotEntryType* plots_test             = 0;
  plotEntryType* plots_test_reweighted1 = 0;
  plotEntryType* plots_test_reweighted2 = 0;
  plotEntryType* plots_test_reweighted3 = 0;
  
  TH2* lut_reweight1 = 0;
  TH2* lut_reweight2 = 0;
  TH2* lut_reweight3 = 0;

  std::string legenEntry_reference = "";
  std::string legenEntry_test      = "";
  
  if ( mode == "genEmbedded" || mode == "recEmbedded" ) {
    plots_reference = new plotEntryType("Ztautau", "");
    plots_reference->fillHistograms(tree_reference, std::vector<weightEntryType*>(), maxEvents);
    
    plots_test = new plotEntryType(mode, "");
    plots_test->fillHistograms(tree_test, std::vector<weightEntryType*>(), maxEvents);

    plots_test_reweighted1 = new plotEntryType(mode, "reweighted1");
    std::cout << "reweighting by muon2Eta_vs_muon1Eta (1):" << std::endl;
    std::cout << " integral(Ztautau) = " << plots_reference->histogramMuon2Eta_vs_muon1Eta_->Integral() << std::endl; 
    std::cout << " integral(Embedded) = " << plots_test->histogramMuon2Eta_vs_muon1Eta_->Integral() << std::endl; 
    lut_reweight1 = dynamic_cast<TH2*>(compRatioHistogram(
      "embeddingKineReweight_muon2Eta_vs_muon1Eta",
      plots_reference->histogramMuon2Eta_vs_muon1Eta_, 
      plots_test->histogramMuon2Eta_vs_muon1Eta_));
    std::vector<weightEntryType*> weightEntries_Embedded_reweighted1;
    weightEntries_Embedded_reweighted1.push_back(new weightEntryType("Embedded_reweighted1", lut_reweight1, kMuon1Eta, kMuon2Eta));
    plots_test_reweighted1->fillHistograms(tree_test, weightEntries_Embedded_reweighted1, maxEvents);

    plots_test_reweighted2 = new plotEntryType(mode, "reweighted2");
    std::cout << "reweighting by diMuonMass_vs_diMuonPt (2):" << std::endl;
    std::cout << " integral(Ztautau) = " << plots_reference->histogramDiMuonMass_vs_diMuonPt_varBinning_->Integral() << std::endl;
    std::cout << " integral(Embedded) = " << plots_test_reweighted1->histogramDiMuonMass_vs_diMuonPt_varBinning_->Integral() << std::endl;
    lut_reweight2 = dynamic_cast<TH2*>(compRatioHistogram(
      "embeddingKineReweight_diMuonMass_vs_diMuonPt", 
      plots_reference->histogramDiMuonMass_vs_diMuonPt_varBinning_, 
      plots_test_reweighted1->histogramDiMuonMass_vs_diMuonPt_varBinning_));
    std::vector<weightEntryType*> weightEntries_Embedded_reweighted2 = weightEntries_Embedded_reweighted1;
    weightEntries_Embedded_reweighted2.push_back(new weightEntryType("Embedded_reweighted2", lut_reweight2, kDiMuonPt, kDiMuonMass));
    plots_test_reweighted2->fillHistograms(tree_test, weightEntries_Embedded_reweighted2, maxEvents);
    
    plots_test_reweighted3 = new plotEntryType(mode, "reweighted3");
    std::cout << "reweighting by muon2Pt_vs_muon1Pt (3):" << std::endl;
    std::cout << " integral(Ztautau) = " << plots_reference->histogramMuon2Pt_vs_muon1Pt_varBinning_->Integral() << std::endl;
    std::cout << " integral(Embedded) = " << plots_test_reweighted2->histogramMuon2Pt_vs_muon1Pt_varBinning_->Integral() << std::endl;
    lut_reweight3 = dynamic_cast<TH2*>(compRatioHistogram(
      "embeddingKineReweight_muon2Pt_vs_muon1Pt",
      plots_reference->histogramMuon2Pt_vs_muon1Pt_varBinning_, 
      plots_test_reweighted2->histogramMuon2Pt_vs_muon1Pt_varBinning_));
    std::vector<weightEntryType*> weightEntries_Embedded_reweighted3 = weightEntries_Embedded_reweighted2;
    weightEntries_Embedded_reweighted3.push_back(new weightEntryType("Embedded_reweighted3", lut_reweight3, kMuon1Pt, kMuon2Pt));
    plots_test_reweighted3->fillHistograms(tree_test, weightEntries_Embedded_reweighted3, maxEvents);
/*  
    plots_test_reweighted1 = new plotEntryType(mode, "reweighted1");
    std::cout << "reweighting by muon2Pt_vs_muon1Pt (1):" << std::endl;
    std::cout << " integral(Ztautau) = " << plots_reference->histogramMuon2Pt_vs_muon1Pt_varBinning_->Integral() << std::endl; 
    std::cout << " integral(Embedded) = " << plots_test->histogramMuon2Pt_vs_muon1Pt_varBinning_->Integral() << std::endl;
    lut_reweight1 = dynamic_cast<TH2*>(compRatioHistogram(
      "embeddingKineReweight_muon2Pt_vs_muon1Pt",
      plots_reference->histogramMuon2Pt_vs_muon1Pt_varBinning_, 
      plots_test->histogramMuon2Pt_vs_muon1Pt_varBinning_));
    std::vector<weightEntryType*> weightEntries_Embedded_reweighted1;
    weightEntries_Embedded_reweighted1.push_back(new weightEntryType("Embedded_reweighted1", lut_reweight1, kMuon1Pt, kMuon2Pt));
    plots_test_reweighted1->fillHistograms(tree_test, weightEntries_Embedded_reweighted1, maxEvents);

    plots_test_reweighted2 = new plotEntryType(mode, "reweighted2");
    std::cout << "reweighting by muon2Eta_vs_muon1Eta (1):" << std::endl;
    std::cout << " integral(Ztautau) = " << plots_reference->histogramMuon2Eta_vs_muon1Eta_->Integral() << std::endl; 
    std::cout << " integral(Embedded) = " << plots_test_reweighted1->histogramMuon2Eta_vs_muon1Eta_->Integral() << std::endl; 
    lut_reweight2 = dynamic_cast<TH2*>(compRatioHistogram(
      "embeddingKineReweight_muon2Eta_vs_muon1Eta",
      plots_reference->histogramMuon2Eta_vs_muon1Eta_, 
      plots_test_reweighted1->histogramMuon2Eta_vs_muon1Eta_));
    std::vector<weightEntryType*> weightEntries_Embedded_reweighted2 = weightEntries_Embedded_reweighted1;
    weightEntries_Embedded_reweighted2.push_back(new weightEntryType("Embedded_reweighted2", lut_reweight2, kMuon1Eta, kMuon2Eta));
    plots_test_reweighted2->fillHistograms(tree_test, weightEntries_Embedded_reweighted2, maxEvents);
    
    plots_test_reweighted3 = new plotEntryType(mode, "reweighted3");
    std::cout << "reweighting by diMuonMass_vs_diMuonPt (3):" << std::endl;
    std::cout << " integral(Ztautau) = " << plots_reference->histogramDiMuonMass_vs_diMuonPt_varBinning_->Integral() << std::endl;
    std::cout << " integral(Embedded) = " << plots_test_reweighted2->histogramDiMuonMass_vs_diMuonPt_varBinning_->Integral() << std::endl; 
    lut_reweight3 = dynamic_cast<TH2*>(compRatioHistogram(
      "embeddingKineReweight_diMuonMass_vs_diMuonPt", 
      plots_reference->histogramDiMuonMass_vs_diMuonPt_varBinning_, 
      plots_test_reweighted2->histogramDiMuonMass_vs_diMuonPt_varBinning_));
    std::vector<weightEntryType*> weightEntries_Embedded_reweighted3 = weightEntries_Embedded_reweighted2;
    weightEntries_Embedded_reweighted3.push_back(new weightEntryType("Embedded_reweighted3", lut_reweight3, kDiMuonPt, kDiMuonMass));
    plots_test_reweighted3->fillHistograms(tree_test, weightEntries_Embedded_reweighted3, maxEvents);
 */     
    legenEntry_reference = "gen. Z/#gamma^{*} #rightarrow #tau #tau";
    if      ( mode == "genEmbedded" ) legenEntry_test = "gen. Embedding";
    else if ( mode == "recEmbedded" ) legenEntry_test = "rec. Embedding";
  } else if ( mode == "Zmumu" ) {
    plots_reference = new plotEntryType("genZmumu", "");
    plots_reference->fillHistograms(tree_reference, std::vector<weightEntryType*>(), maxEvents);
    
    plots_test = new plotEntryType("recZmumu", "");
    plots_test->fillHistograms(tree_test, std::vector<weightEntryType*>(), maxEvents);
    
    plots_test_reweighted1 = new plotEntryType("recZmumu", "reweighted1");
    std::cout << "reweighting by muon2Pt_vs_muon1Pt (1):" << std::endl;
    std::cout << " integral(genZmumu) = " << plots_reference->histogramMuon2Pt_vs_muon1Pt_varBinning_->Integral() << std::endl;
    std::cout << " integral(recZmumu) = " << plots_test->histogramMuon2Pt_vs_muon1Pt_varBinning_->Integral() << std::endl;
    lut_reweight1 = dynamic_cast<TH2*>(compRatioHistogram(
      "embeddingKineReweight_muon2Pt_vs_muon1Pt",
      plots_reference->histogramMuon2Pt_vs_muon1Pt_varBinning_, 
      plots_test->histogramMuon2Pt_vs_muon1Pt_varBinning_));
    std::vector<weightEntryType*> weightEntries_Embedded_reweighted1;
    weightEntries_Embedded_reweighted1.push_back(new weightEntryType("Embedded_reweighted1", lut_reweight1, kMuon1Pt, kMuon2Pt)); 
    plots_test_reweighted1->fillHistograms(tree_test, weightEntries_Embedded_reweighted1, maxEvents);
    
    plots_test_reweighted2 = new plotEntryType("recZmumu", "reweighted2");
    std::cout << "reweighting by diMuonMass_vs_diMuonPt (2):" << std::endl;
    std::cout << " integral(genZmumu) = " << plots_reference->histogramDiMuonMass_vs_diMuonPt_varBinning_->Integral() << std::endl;
    std::cout << " integral(recZmumu) = " << plots_test_reweighted1->histogramDiMuonMass_vs_diMuonPt_varBinning_->Integral() << std::endl;
    lut_reweight2 = dynamic_cast<TH2*>(compRatioHistogram(
      "embeddingKineReweight_diMuonMass_vs_diMuonPt", 
      plots_reference->histogramDiMuonMass_vs_diMuonPt_varBinning_, 
      plots_test_reweighted1->histogramDiMuonMass_vs_diMuonPt_varBinning_));
    std::vector<weightEntryType*> weightEntries_Embedded_reweighted2 = weightEntries_Embedded_reweighted1;
    weightEntries_Embedded_reweighted2.push_back(new weightEntryType("Embedded_reweighted2", lut_reweight2, kDiMuonPt, kDiMuonMass));
    plots_test_reweighted2->fillHistograms(tree_test, weightEntries_Embedded_reweighted2, maxEvents);
    
    legenEntry_reference = "gen. Z/#gamma^{*} #rightarrow #mu #mu";
    legenEntry_test      = "rec. Z/#gamma^{*} #rightarrow #mu #mu";
  } else assert(0);
  
  std::string legenEntry_test_reweighted1 = Form("%s, reweighted (1)", legenEntry_test.data());
  std::string legenEntry_test_reweighted2 = Form("%s, reweighted (2)", legenEntry_test.data());
  std::string legenEntry_test_reweighted3 = Form("%s, reweighted (3)", legenEntry_test.data());

  showDistribution(800, 900,
		   plots_reference->histogramMuon1Pt_fixedBinning_, legenEntry_reference,
		   plots_test->histogramMuon1Pt_fixedBinning_, legenEntry_test,
		   plots_test_reweighted1->histogramMuon1Pt_fixedBinning_, legenEntry_test_reweighted1,
		   plots_test_reweighted2->histogramMuon1Pt_fixedBinning_, legenEntry_test_reweighted2,
		   plots_test_reweighted3->histogramMuon1Pt_fixedBinning_, legenEntry_test_reweighted3,
		   //0, "",
		   0, "",
		   0., 250., 50, "P_{T}^{1} / GeV", 1.3,
		   true, 1.e-5, 1.e+1, "a.u", 1.3,
		   0.49, 0.71,
		   Form("plots/makeEmbeddingKineReweightLUTs2_%s_%s_muon1Pt.pdf", channel.data(), mode.data()));
  showDistribution(800, 900,
		   plots_reference->histogramMuon1Eta_, legenEntry_reference,
		   plots_test->histogramMuon1Eta_, legenEntry_test,
		   plots_test_reweighted1->histogramMuon1Eta_, legenEntry_test_reweighted1,
		   plots_test_reweighted2->histogramMuon1Eta_, legenEntry_test_reweighted2,
		   plots_test_reweighted3->histogramMuon1Eta_, legenEntry_test_reweighted3,
		   //0, "",
		   0, "",
		   -2.5, +2.5, 50, "#eta_{1}", 1.3,
		   true, 1.e-5, 1.e+1, "a.u", 1.3,
		   0.49, 0.71,
		   Form("plots/makeEmbeddingKineReweightLUTs2_%s_%s_muon1Eta.pdf", channel.data(), mode.data()));
  showDistribution(800, 900,
		   plots_reference->histogramMuon2Pt_fixedBinning_, legenEntry_reference,
		   plots_test->histogramMuon2Pt_fixedBinning_, legenEntry_test,
		   plots_test_reweighted1->histogramMuon2Pt_fixedBinning_, legenEntry_test_reweighted1,
		   plots_test_reweighted2->histogramMuon2Pt_fixedBinning_, legenEntry_test_reweighted2,
		   plots_test_reweighted3->histogramMuon2Pt_fixedBinning_, legenEntry_test_reweighted3,
		   //0, "",
		   0, "",
		   0., 250., 50, "P_{T}^{2} / GeV", 1.3,
		   true, 1.e-5, 1.e+1, "a.u", 1.3,
		   0.49, 0.71,
		   Form("plots/makeEmbeddingKineReweightLUTs2_%s_%s_muon2Pt.pdf", channel.data(), mode.data()));
  showDistribution(800, 900,
		   plots_reference->histogramMuon2Eta_, legenEntry_reference,
		   plots_test->histogramMuon2Eta_, legenEntry_test,
		   plots_test_reweighted1->histogramMuon2Eta_, legenEntry_test_reweighted1,
		   plots_test_reweighted2->histogramMuon2Eta_, legenEntry_test_reweighted2,
		   plots_test_reweighted3->histogramMuon2Eta_, legenEntry_test_reweighted3,
		   //0, "",
		   0, "",
		   -2.5, +2.5, 50, "#eta_{2}", 1.3,
		   true, 1.e-5, 1.e+1, "a.u", 1.3,
		   0.49, 0.71,
		   Form("plots/makeEmbeddingKineReweightLUTs2_%s_%s_muon2Eta.pdf", channel.data(), mode.data()));
  showDistribution(800, 900,
		   plots_reference->histogramDiMuonPt_fixedBinning_, legenEntry_reference,
		   plots_test->histogramDiMuonPt_fixedBinning_, legenEntry_test,
		   plots_test_reweighted1->histogramDiMuonPt_fixedBinning_, legenEntry_test_reweighted1,
		   plots_test_reweighted2->histogramDiMuonPt_fixedBinning_, legenEntry_test_reweighted2,
		   plots_test_reweighted3->histogramDiMuonPt_fixedBinning_, legenEntry_test_reweighted3,
		   //0, "",
		   0, "",
		   0., 250., 50, "P_{T}^{#mu#mu} / GeV", 1.3, 
		   true, 1.e-5, 1.e+1, "a.u", 1.3,
		   0.49, 0.71,
		   Form("plots/makeEmbeddingKineReweightLUTs2_%s_%s_diMuonPt.pdf", channel.data(), mode.data()));
  showDistribution(800, 900,
		   plots_reference->histogramDiMuonMass_fixedBinning_, legenEntry_reference,
		   plots_test->histogramDiMuonMass_fixedBinning_, legenEntry_test,
		   plots_test_reweighted1->histogramDiMuonMass_fixedBinning_, legenEntry_test_reweighted1,
		   plots_test_reweighted2->histogramDiMuonMass_fixedBinning_, legenEntry_test_reweighted2,
		   plots_test_reweighted3->histogramDiMuonMass_fixedBinning_, legenEntry_test_reweighted3,
		   //0, "",
		   0, "",
		   0., 250., 50, "M_{#mu#mu} / GeV", 1.3,
		   true, 1.e-5, 1.e+1, "a.u", 1.3,
		   0.49, 0.71,
		   Form("plots/makeEmbeddingKineReweightLUTs2_%s_%s_diMuonMass.pdf", channel.data(), mode.data()));
  showDistribution(800, 900,
		   plots_reference->histogramDiMuonMass_fixedBinning_, legenEntry_reference,
		   plots_test->histogramDiMuonMass_fixedBinning_, legenEntry_test,
		   plots_test_reweighted1->histogramDiMuonMass_fixedBinning_, legenEntry_test_reweighted1,
		   plots_test_reweighted2->histogramDiMuonMass_fixedBinning_, legenEntry_test_reweighted2,
		   plots_test_reweighted3->histogramDiMuonMass_fixedBinning_, legenEntry_test_reweighted3,
		   //0, "",
		   0, "",
		   0., 250., 50, "M_{#mu#mu} / GeV", 1.3,
		   true, 1.e-5, 1.e+1, "a.u", 1.3,
		   0.49, 0.71,
		   Form("plots/makeEmbeddingKineReweightLUTs2_%s_%s_diMuonMass.pdf", channel.data(), mode.data()));
  showDistribution(800, 900,
		   plots_reference->histogramNumJetsCorrPtGt20_, legenEntry_reference,
		   plots_test->histogramNumJetsCorrPtGt20_, legenEntry_test,
		   plots_test_reweighted1->histogramNumJetsCorrPtGt20_, legenEntry_test_reweighted1,
		   plots_test_reweighted2->histogramNumJetsCorrPtGt20_, legenEntry_test_reweighted2,
		   plots_test_reweighted3->histogramNumJetsCorrPtGt20_, legenEntry_test_reweighted3,
		   //0, "",
		   0, "",
		   -0.5, 19.5, 20, "N_{jet} (P_{T}^{corr} > 20 GeV)", 1.3,
		   true, 1.e-5, 1.e+1, "a.u", 1.3,
		   0.49, 0.71,
		   Form("plots/makeEmbeddingKineReweightLUTs2_%s_%s_numJetsCorrPtGt20.pdf", channel.data(), mode.data()));
  showDistribution(800, 900,
		   plots_reference->histogramNumJetsCorrPtGt30_, legenEntry_reference,
		   plots_test->histogramNumJetsCorrPtGt30_, legenEntry_test,
		   plots_test_reweighted1->histogramNumJetsCorrPtGt30_, legenEntry_test_reweighted1,
		   plots_test_reweighted2->histogramNumJetsCorrPtGt30_, legenEntry_test_reweighted2,
		   plots_test_reweighted3->histogramNumJetsCorrPtGt30_, legenEntry_test_reweighted3,
		   //0, "",
		   0, "",
		   -0.5, 19.5, 20, "N_{jet} (P_{T}^{corr} > 30 GeV)", 1.3,
		   true, 1.e-5, 1.e+1, "a.u", 1.3,
		   0.49, 0.71,
		   Form("plots/makeEmbeddingKineReweightLUTs2_%s_%s_numJetsCorrPtGt30.pdf", channel.data(), mode.data()));
  showDistribution(800, 900,
		   plots_reference->histogramPFMEt_, legenEntry_reference,
		   plots_test->histogramPFMEt_, legenEntry_test,
		   plots_test_reweighted1->histogramPFMEt_, legenEntry_test_reweighted1,
		   plots_test_reweighted2->histogramPFMEt_, legenEntry_test_reweighted2,
		   plots_test_reweighted3->histogramPFMEt_, legenEntry_test_reweighted3,
		   //0, "",
		   0, "",
		   0., 250., 50, "pfMEt / GeV", 1.3,
		   true, 1.e-5, 1.e+1, "a.u", 1.3,
		   0.49, 0.71,
		   Form("plots/makeEmbeddingKineReweightLUTs2_%s_%s_pfMEt.pdf", channel.data(), mode.data()));
  showDistribution(800, 900,
		   plots_reference->histogramCaloMEt_, legenEntry_reference,
		   plots_test->histogramCaloMEt_, legenEntry_test,
		   plots_test_reweighted1->histogramCaloMEt_, legenEntry_test_reweighted1,
		   plots_test_reweighted2->histogramCaloMEt_, legenEntry_test_reweighted2,
		   plots_test_reweighted3->histogramCaloMEt_, legenEntry_test_reweighted3,
		   //0, "",
		   0, "",
		   0., 250., 50, "caloMEt / GeV", 1.3,
		   true, 1.e-5, 1.e+1, "a.u", 1.3,
		   0.49, 0.71,
		   Form("plots/makeEmbeddingKineReweightLUTs2_%s_%s_caloMEt.pdf", channel.data(), mode.data()));

  TString outputFileName = Form("makeEmbeddingKineReweightLUTs2_%s_%s.root", channel.data(), mode.data());
  TFile* outputFile = TFile::Open(outputFileName.Data(), "RECREATE");
  //plots_reference->saveHistograms();
  //plots_test->saveHistograms();
  //plots_test_reweighted1->saveHistograms();
  //plots_test_reweighted2->saveHistograms();
  lut_reweight1->Write();
  lut_reweight2->Write();
  lut_reweight3->Write();
  delete outputFile;

  delete inputFile_reference;
  if ( inputFile_test != inputFile_reference ) delete inputFile_test;
}




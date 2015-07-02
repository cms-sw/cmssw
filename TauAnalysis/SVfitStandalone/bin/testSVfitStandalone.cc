
/**
   \class testSVfitStandalone testSVfitStandalone.cc "TauAnalysis/SVfitStandalone/bin/testSVfitStandalone.cc"
   \brief Basic example of the use of the standalone version of SVfit

   This is an example executable to show the use of the standalone version of SVfit 
   from a flat n-tuple or single event.
*/

#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "TauAnalysis/SVfitStandalone/interface/SVfitStandaloneAlgorithm.h"

#include "TFile.h"
#include "TTree.h"
#include "TH1.h"

void singleEvent()
{
  /* 
     This is a single event for testing in the integration mode.
  */
  // define MET
  double measuredMETx =  11.7491;
  double measuredMETy = -51.9172; 
  // define MET covariance
  TMatrixD covMET(2, 2);
  covMET[0][0] =  787.352;
  covMET[1][0] = -178.63;
  covMET[0][1] = -178.63;
  covMET[1][1] =  179.545;
  // define lepton four vectors
  std::vector<svFitStandalone::MeasuredTauLepton> measuredTauLeptons;
  measuredTauLeptons.push_back(svFitStandalone::MeasuredTauLepton(svFitStandalone::kTauToElecDecay, 33.7393, 0.9409,  -0.541458, 0.51100e-3)); // tau -> electron decay (Pt, eta, phi, mass)
  measuredTauLeptons.push_back(svFitStandalone::MeasuredTauLepton(svFitStandalone::kTauToHadDecay,  25.7322, 0.618228, 2.79362,  0.13957, 0)); // tau -> 1prong0pi0 hadronic decay (Pt, eta, phi, mass)
  // define algorithm (set the debug level to 3 for testing)
  unsigned verbosity = 2;
  SVfitStandaloneAlgorithm algo(measuredTauLeptons, measuredMETx, measuredMETy, covMET, verbosity);
  algo.addLogM(false);  
  edm::FileInPath inputFileName_visPtResolution("TauAnalysis/SVfitStandalone/data/svFitVisMassAndPtResolutionPDF.root");
  TH1::AddDirectory(false);  
  TFile* inputFile_visPtResolution = new TFile(inputFileName_visPtResolution.fullPath().data());
  algo.shiftVisPt(true, inputFile_visPtResolution);
  /* 
     the following lines show how to use the different methods on a single event
  */
  // minuit fit method
  //algo.fit();
  // integration by VEGAS (same as function algo.integrate() that has been in use when markov chain integration had not yet been implemented)
  //algo.integrateVEGAS();
  // integration by markov chain MC
  algo.integrateMarkovChain();

  double mass = algo.getMass(); // return value is in units of GeV
  if ( algo.isValidSolution() ) {
    std::cout << "found mass = " << mass << " (expected value = 118.64)" << std::endl;
  } else {
    std::cout << "sorry -- status of NLL is not valid [" << algo.isValidSolution() << "]" << std::endl;
  }

  delete inputFile_visPtResolution;
}

void eventsFromTree(int argc, char* argv[]) 
{
  // parse arguments
  if ( argc < 3 ) {
    std::cout << "Usage : " << argv[0] << " [inputfile.root] [tree_name]" << std::endl;
    return;
  }
  // get intput directory up to one before mass points
  TFile* file = new TFile(argv[1]); 
  // access tree in file
  TTree* tree = (TTree*) file->Get(argv[2]);
  // input variables
  float met, metPhi;
  float covMet11, covMet12; 
  float covMet21, covMet22;
  float l1Pt, l1Eta, l1Phi, l1Mass;
  float l2Pt, l2Eta, l2Phi, l2Mass;
  float mTrue;
  // branch adresses
  tree->SetBranchAddress("met", &met);
  tree->SetBranchAddress("mphi", &metPhi);
  tree->SetBranchAddress("mcov_11", &covMet11);
  tree->SetBranchAddress("mcov_12", &covMet12);
  tree->SetBranchAddress("mcov_21", &covMet21);
  tree->SetBranchAddress("mcov_22", &covMet22);
  tree->SetBranchAddress("l1_Pt", &l1Pt);
  tree->SetBranchAddress("l1_Eta", &l1Eta);
  tree->SetBranchAddress("l1_Phi", &l1Phi);
  tree->SetBranchAddress("l1_M", &l1Mass);
  tree->SetBranchAddress("l2_Pt", &l2Pt);
  tree->SetBranchAddress("l2_Eta", &l2Eta);
  tree->SetBranchAddress("l2_Phi", &l2Phi);
  tree->SetBranchAddress("l2_M", &l2Mass);
  tree->SetBranchAddress("m_true", &mTrue);
  int nevent = tree->GetEntries();
  for ( int i = 0; i < nevent; ++i ) {
    tree->GetEvent(i);
    std::cout << "event " << (i + 1) << std::endl;
    // setup MET input vector
    double measuredMETx = met*TMath::Cos(metPhi);
    double measuredMETy = met*TMath::Sin(metPhi);
    // setup the MET significance
    TMatrixD covMET(2,2);
    covMET[0][0] = covMet11;
    covMET[0][1] = covMet12;
    covMET[1][0] = covMet21;
    covMET[1][1] = covMet22;
    // setup measure tau lepton vectors 
    svFitStandalone::kDecayType l1Type, l2Type;
    if ( std::string(argv[2]) == "EMu" ) {
      l1Type = svFitStandalone::kTauToElecDecay;
      l2Type = svFitStandalone::kTauToMuDecay;
    } else if ( std::string(argv[2]) == "MuTau" ) {
      l1Type = svFitStandalone::kTauToMuDecay;
      l2Type = svFitStandalone::kTauToHadDecay;
    } else if ( std::string(argv[2]) == "ETau" ) {
      l1Type = svFitStandalone::kTauToElecDecay;
      l2Type = svFitStandalone::kTauToHadDecay;
    } else if ( std::string(argv[2]) == "TauTau" ) {
      l1Type = svFitStandalone::kTauToHadDecay;
      l2Type = svFitStandalone::kTauToHadDecay;
    } else {
      std::cerr << "Error: Invalid channel = " << std::string(argv[2]) << " !!" << std::endl;
      std::cerr << "(some customization of this code will be needed for your analysis)" << std::endl;
      assert(0);
    }
    std::vector<svFitStandalone::MeasuredTauLepton> measuredTauLeptons;
    measuredTauLeptons.push_back(svFitStandalone::MeasuredTauLepton(l1Type, l1Pt, l1Eta, l1Phi, l1Mass));
    measuredTauLeptons.push_back(svFitStandalone::MeasuredTauLepton(l2Type, l2Pt, l2Eta, l2Phi, l2Mass));
    // construct the class object from the minimal necesarry information
    SVfitStandaloneAlgorithm algo(measuredTauLeptons, measuredMETx, measuredMETy, covMET, 1);
    // apply customized configurations if wanted (examples are given below)
    algo.maxObjFunctionCalls(5000);
    //algo.addLogM(false);
    //algo.metPower(0.5)
    // minuit fit method
    //algo.fit();
    // integration by VEGAS (default)
    algo.integrateVEGAS();
    // integration by markov chain MC
    //algo.integrateMarkovChain();
    // retrieve the results upon success
    std::cout << "... m truth : " << mTrue << std::endl;
    if ( algo.isValidSolution() ) {
      std::cout << "... m svfit : " << algo.mass() << " +/- " << algo.massUncert() << std::endl; // return value is in units of GeV
    } else {
      std::cout << "... m svfit : ---" << std::endl;
    }
  }
  return;
}

int main(int argc, char* argv[]) 
{
  //eventsFromTree(argc, argv);
  singleEvent();
  return 0;
}
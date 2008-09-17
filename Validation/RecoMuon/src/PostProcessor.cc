/*
 *  Class:PostProcessor 
 *
 *
 *  $Date: 2008/07/15 18:43:44 $
 *  $Revision: 1.3 $
 * 
 *  \author Junghwan Goh - SungKyunKwan University
 */

#include "Validation/RecoMuon/src/PostProcessor.h"

#include "Validation/Tools/interface/FitSlicesYTool.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <TH1F.h>
#include <cmath>
//#include <boost/tokenizer.hpp>

using namespace std;
using namespace edm;

typedef MonitorElement ME;
typedef vector<string> vstring;

PostProcessor::PostProcessor(const ParameterSet& pset)
{
  commands_ = pset.getParameter<vstring>("commands");
  outputFileName_ = pset.getUntrackedParameter<string>("outputFileName", "");
  subDir_ = pset.getUntrackedParameter<string>("subDir");
}

void PostProcessor::endJob()
{
  DQMStore * dqm = 0;
  dqm = Service<DQMStore>().operator->();

  if ( ! dqm ) {
    LogInfo("PostProcessor") << "Cannot create DQMStore instance\n";
    return;
  }

  theDQM = dqm;
  
  for(vstring::const_iterator iCmd = commands_.begin();
      iCmd != commands_.end(); ++iCmd) {
    const string& cmd = *iCmd;

    // Parse a command using boost::tokenizer
    using namespace boost;
    typedef escaped_list_separator<char> elsc;
    tokenizer<elsc> tokens(cmd, elsc("\\", " \t", "\'"));

    vector<tokenizer<elsc>::value_type> args;

    for(tokenizer<elsc>::const_iterator tok_iter = tokens.begin();
        tok_iter != tokens.end(); ++tok_iter) {
      args.push_back(*tok_iter);
    }

    if ( args.empty() ) continue;

    processLoop(args[0],args);

  }

  if ( ! outputFileName_.empty() ) theDQM->save(outputFileName_);
}

void PostProcessor::computeEfficiency(const string& startDir, const string& efficMEName, const string& efficMETitle,
                                      const string& recoMEName, const string& simMEName)
{
  theDQM->cd(startDir);
  ME* simME  = theDQM->get(simMEName);
  ME* recoME = theDQM->get(recoMEName);
  if ( !simME || !recoME ) {
    LogError("PostProcessor") << "computeEfficiency() : No reco-ME '" << recoMEName 
                              << "' or sim-ME '" << simMEName << "' found\n";
    return;
  }

  TH1F* hSim  = simME ->getTH1F();
  TH1F* hReco = recoME->getTH1F();
  if ( !hSim || !hReco ) {
    LogError("PostProcessor") << "computeEfficiency() : Cannot create TH1F from ME\n";
    return;
  }

  theDQM->setCurrentFolder(startDir);
  ME* efficME = theDQM->book1D(efficMEName, efficMETitle, hSim->GetNbinsX(), hSim->GetXaxis()->GetXmin(), hSim->GetXaxis()->GetXmax());
  if ( !efficME ) {
    LogError("PostProcessor") << "computeEfficiency() : Cannot book effic-ME from the DQM\n";
    return;
  }

  hReco->Sumw2();
  hSim->Sumw2();
  //  efficME->getTH1F()->Divide(hReco, hSim, 1., 1., "B");

  const int nBin = efficME->getNbinsX();
  for(int bin = 0; bin <= nBin; ++bin) {
    const float nSim  = simME ->getBinContent(bin);
    const float nReco = recoME->getBinContent(bin);
    const float eff = nSim ? nReco/nSim : 0.;
    const float err = nSim && eff <= 1 ? sqrt(eff*(1-eff)/nSim) : 0.;
    efficME->setBinContent(bin, eff);
    efficME->setBinError(bin, err);
  }

}

void PostProcessor::computeResolution(const string& startDir, const string& namePrefix, const string& titlePrefix,
                                      const std::string& srcName)
{
  theDQM->cd(startDir);
  ME* srcME = theDQM->get(srcName);
  if ( !srcME ) {
    LogError("PostProcessor") << "computeResolution() : No source ME '" << srcName << "' found\n";
    return;
  }

  TH2F* hSrc = srcME->getTH2F();
  if ( !hSrc ) {
    LogError("PostProcessor") << "computeResolution() : Cannot create TH2F from source-ME\n";
    return;
  }

  theDQM->setCurrentFolder(startDir);

  const int nBin = hSrc->GetNbinsX();
  const double xMin = hSrc->GetXaxis()->GetXmin();
  const double xMax = hSrc->GetXaxis()->GetXmax();

  ME* meanME = theDQM->book1D(namePrefix+"_Mean", titlePrefix+" Mean", nBin, xMin, xMax);
  ME* sigmaME = theDQM->book1D(namePrefix+"_Sigma", titlePrefix+" Sigma", nBin, xMin, xMax);
//  ME* chi2ME  = theDQM->book1D(namePrefix+"_Chi2" , titlePrefix+" #Chi^{2}", nBin, xMin, xMax); // N/A

  FitSlicesYTool fitTool(srcME);
  fitTool.getFittedMeanWithError(meanME);
  fitTool.getFittedSigmaWithError(sigmaME);
//  fitTool.getFittedChisqWithError(chi2ME); // N/A

}

void PostProcessor::processLoop( const std::string& startDir, vector<boost::tokenizer<elsc>::value_type> args) 
{
  if(theDQM->dirExists(startDir)) theDQM->cd(startDir);
  
  std::vector<std::string> subDirs =   theDQM->getSubdirs();
  std::vector<std::string> mes     =  theDQM->getMEs();

  /*
  std::cout << " ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n";

  std::cout << " ------------------------------------------------------------\n"
	    << "                    Current Directory:                   \n"
	    << theDQM->pwd() << std::endl
	    << " ------------------------------------------------------------\n";

  std::cout << " ------------------------------------------------------------\n"
	    << "                    SubDirs:                     \n"
	    << " ------------------------------------------------------------\n";
  
  std::copy(subDirs.begin(), subDirs.end(),
	    std::ostream_iterator<std::string>(std::cout, "\n"));
  
  std::cout << " ------------------------------------------------------------\n";

  std::cout << " ------------------------------------------------------------\n"
	    << "                    MEs:                     \n"
	    << " ------------------------------------------------------------\n";
  
  std::copy(mes.begin(), mes.end(),
	    std::ostream_iterator<std::string>(std::cout, "\n"));
  
  std::cout << " ------------------------------------------------------------\n";
  std::cout << " ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n";
  */

  for(vstring::const_iterator iDir = subDirs.begin(); iDir != subDirs.end();++iDir) {
    const string& theSubDir = *iDir;
    processLoop(theSubDir,args);

  }

  string path1, path2;
    
  switch ( args[1][0] ) {
    // Efficiency plots
  case 'E':
  case 'e':
    if ( args.size() != 6 ) break;;
    path1.clear();
    path1 += startDir;
    path1 += "/";
    path1 += args[4];
    path2.clear();
    path2 += startDir;
    path2 += "/";
    path2 += args[5];
    computeEfficiency(startDir,args[2], args[3], path1, path2);
    break;
    // Resolution plots
  case 'R':
  case 'r':
    if ( args.size() != 5 ) break;;
    path1.clear();
    path1 += startDir;
    path1 += "/";
    path1 += args[4];
    computeResolution(startDir,args[2], args[3], path1);
    break;
  default:
    LogError("PostProcessor") << "Invalid command\n";
  }
  
  
}

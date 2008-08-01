/*
 *  Class:PostProcessor 
 *
 *
 *  $Date: 2008/05/27 14:12:35 $
 *  $Revision: 1.1 $
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
#include <boost/tokenizer.hpp>

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
  theDQM->setCurrentFolder(subDir_);

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

    switch ( args[0][0] ) {
     // Efficiency plots
     case 'E':
     case 'e':
      if ( args.size() != 5 ) continue;;
      computeEfficiency(args[1], args[2], args[3], args[4]);
      break;
     // Resolution plots
     case 'R':
     case 'r':
      if ( args.size() != 4 ) continue;;
      computeResolution(args[1], args[2], args[3]);
      break;
     default:
      LogError("PostProcessor") << "Invalid command\n";
    }
  }

  if ( ! outputFileName_.empty() ) theDQM->save(outputFileName_);
}

void PostProcessor::computeEfficiency(const string& efficMEName, const string& efficMETitle,
                                      const string& recoMEName, const string& simMEName)
{
  ME* simME  = theDQM->get(simMEName );
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

  theDQM->setCurrentFolder(subDir_);
  ME* efficME = theDQM->book1D(efficMEName, efficMETitle, hSim->GetNbinsX(), hSim->GetXaxis()->GetXmin(), hSim->GetXaxis()->GetXmax());
  if ( !efficME ) {
    LogError("PostProcessor") << "computeEfficiency() : Cannot book effic-ME from the DQM\n";
    return;
  }

  hReco->Sumw2();
  hSim->Sumw2();
  efficME->getTH1F()->Divide(hReco, hSim, 1., 1., "B");
/*
  const int nBin = efficME->getNbinsX();
  for(int bin = 0; bin <= nBin; ++bin) {
    const float nSim  = simME ->getBinContent(bin);
    const float nReco = recoME->getBinContent(bin);
    const float eff = nSim ? nReco/nSim : 0.;
    const float err = nSim && eff <= 1 ? sqrt(eff*(1-eff)/nSim) : 0.;
    efficME->setBinContent(bin, eff);
    efficME->setBinError(bin, err);
  }
*/
}

void PostProcessor::computeResolution(const string& namePrefix, const string& titlePrefix,
                                      const std::string& srcName)
{
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

  theDQM->setCurrentFolder(subDir_);
  ME* sigmaME = theDQM->book1D(namePrefix+"_Sigma", titlePrefix+" Sigma", 
                               hSrc->GetNbinsX(), hSrc->GetXaxis()->GetXmin(), hSrc->GetXaxis()->GetXmax());
  FitSlicesYTool fitTool(srcME);
  fitTool.getFittedSigmaWithError(sigmaME);

}


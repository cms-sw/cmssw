#include "TauAnalysis/MCEmbeddingTools/plugins/MuonRadiationCorrWeightProducer.h"

#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "DataFormats/Candidate/interface/Particle.h"

#include "DataFormats/Math/interface/deltaR.h"

#include "TauAnalysis/MCEmbeddingTools/interface/embeddingAuxFunctions.h"

#include <vector>
#include <algorithm>
#include <math.h>

MuonRadiationCorrWeightProducer::MuonRadiationCorrWeightProducer(const edm::ParameterSet& cfg)
  : numOthers_(0),
    numWarnings_(0)
{
  srcMuonsBeforeRad_ = cfg.getParameter<edm::InputTag>("srcMuonsBeforeRad"); 
  srcMuonsAfterRad_ = cfg.getParameter<edm::InputTag>("srcMuonsAfterRad"); 

  edm::FileInPath inputFileName = cfg.getParameter<edm::FileInPath>("inputFileName");
  if ( inputFileName.location() == edm::FileInPath::Unknown)
    throw cms::Exception("MuonRadiationCorrWeightProducer") 
      << " Failed to find File = " << inputFileName << " !!\n";
  std::auto_ptr<TFile> inputFile(new TFile(inputFileName.fullPath().data()));

  typedef std::vector<double> vdouble;
  vdouble binningMuonEn = cfg.getParameter<vdouble>("binningMuonEn");
  int numBinsMuonEn = binningMuonEn.size() - 1;
  if ( !(numBinsMuonEn >= 1) ) throw cms::Exception("Configuration")
    << " Invalid Configuration Parameter 'binningMuonEn', must define at least one bin !!\n";

  lutDirectoryRef_ = cfg.getParameter<std::string>("lutDirectoryRef");
  for ( int iBinMuPlusEn = 0; iBinMuPlusEn < numBinsMuonEn; ++iBinMuPlusEn ) {
    double minMuPlusEn = binningMuonEn[iBinMuPlusEn];
    double maxMuPlusEn = binningMuonEn[iBinMuPlusEn + 1];
    for ( int iBinMuMinusEn = 0; iBinMuMinusEn < numBinsMuonEn; ++iBinMuMinusEn ) {
      double minMuMinusEn = binningMuonEn[iBinMuMinusEn];
      double maxMuMinusEn = binningMuonEn[iBinMuMinusEn + 1];
      lutEntryType* lutEntry = 
        new lutEntryType(*inputFile, lutDirectoryRef_, minMuPlusEn, maxMuPlusEn, minMuMinusEn, maxMuMinusEn);
      lutEntriesRef_.push_back(lutEntry);
    }
  }
  
  edm::ParameterSet cfgNameOthers = cfg.getParameter<edm::ParameterSet>("lutDirectoryOthers");
  std::vector<std::string> names_others = cfgNameOthers.getParameterNamesForType<std::string>();
  for ( std::vector<std::string>::const_iterator name_others = names_others.begin(); 
        name_others != names_others.end(); ++name_others ) {
    std::string lutDirectoryOther = cfgNameOthers.getParameter<std::string>(*name_others);
    lutDirectoriesOthers_[*name_others] = lutDirectoryOther;
    for ( int iBinMuPlusEn = 0; iBinMuPlusEn < numBinsMuonEn; ++iBinMuPlusEn ) {
      double minMuPlusEn = binningMuonEn[iBinMuPlusEn];
      double maxMuPlusEn = binningMuonEn[iBinMuPlusEn + 1];
      for ( int iBinMuMinusEn = 0; iBinMuMinusEn < numBinsMuonEn; ++iBinMuMinusEn ) {
	double minMuMinusEn = binningMuonEn[iBinMuMinusEn];
	double maxMuMinusEn = binningMuonEn[iBinMuMinusEn + 1];
	lutEntryType* lutEntry = 
          new lutEntryType(*inputFile, lutDirectoryOther, minMuPlusEn, maxMuPlusEn, minMuMinusEn, maxMuMinusEn);
        lutEntriesOthers_[*name_others].push_back(lutEntry);
      }
    }    
    ++numOthers_;
  }
  if ( !(numOthers_ >= 1) )
    throw cms::Exception("MuonRadiationCorrWeightProducer") 
      << " Invalid Configuration Parameter 'lutNameOthers': No alternative models defined !!\n";
  
  minWeight_ = cfg.getParameter<double>("minWeight"); 
  maxWeight_ = cfg.getParameter<double>("maxWeight");

  verbosity_ = ( cfg.exists("verbosity") ) ?
    cfg.getParameter<int>("verbosity") : 0;

  maxWarnings_ = 10;

  produces<double>("weight");
  produces<double>("weightUp");
  produces<double>("weightDown");
}

MuonRadiationCorrWeightProducer::~MuonRadiationCorrWeightProducer() 
{
  for ( vlutEntryType::iterator it = lutEntriesRef_.begin();
	it != lutEntriesRef_.end(); ++it ) {
    delete (*it);
  }
  
  for ( std::map<std::string, vlutEntryType>::iterator it1 = lutEntriesOthers_.begin();
	it1 != lutEntriesOthers_.end(); ++it1 ) {
    for ( vlutEntryType::iterator it2 = it1->second.begin();
	  it2 != it1->second.end(); ++it2 ) {
      delete (*it2);
    }
  }
}

namespace
{
  double square(double x)
  {
    return x*x;
  }
}

void MuonRadiationCorrWeightProducer::produce(edm::Event& evt, const edm::EventSetup& es)
{
  if ( verbosity_ ) {
    std::cout << "<MuonRadiationCorrWeightProducer::produce>:" << std::endl;
    std::cout << " srcMuonsBeforeRad = " << srcMuonsBeforeRad_ << std::endl;
    std::cout << " srcMuonsAfterRad = " << srcMuonsAfterRad_ << std::endl;
  }

  reco::Candidate::LorentzVector genMuonPlusP4_beforeRad;
  bool genMuonPlus_beforeRad_found = false;
  reco::Candidate::LorentzVector genMuonMinusP4_beforeRad;
  bool genMuonMinus_beforeRad_found = false;
  findMuons(evt, srcMuonsBeforeRad_, genMuonPlusP4_beforeRad, genMuonPlus_beforeRad_found, genMuonMinusP4_beforeRad, genMuonMinus_beforeRad_found);

  reco::Candidate::LorentzVector genMuonPlusP4_afterRad;
  bool genMuonPlus_afterRad_found = false;
  reco::Candidate::LorentzVector genMuonMinusP4_afterRad;
  bool genMuonMinus_afterRad_found = false;
  findMuons(evt, srcMuonsAfterRad_, genMuonPlusP4_afterRad, genMuonPlus_afterRad_found, genMuonMinusP4_afterRad, genMuonMinus_afterRad_found);

  bool genMuonPlus_found = (genMuonPlus_beforeRad_found && genMuonPlus_afterRad_found);
  bool genMuonMinus_found = (genMuonMinus_beforeRad_found && genMuonMinus_afterRad_found);
  
  double weight = 1.;
  double weightErr = 1.;
  if ( genMuonPlus_found && genMuonMinus_found ) {
    if ( verbosity_ ) {
      std::cout << " muon+: En = " << genMuonPlusP4_afterRad.E() << ", dEn = " << (genMuonPlusP4_beforeRad.E() - genMuonPlusP4_afterRad.E()) << std::endl;
      std::cout << " muon-: En = " << genMuonMinusP4_afterRad.E() << ", dEn = " << (genMuonMinusP4_beforeRad.E() - genMuonMinusP4_afterRad.E()) << std::endl;
    }
    
    int error = 0;

    double pRef = 0;
    bool pRef_found = false;
    for ( vlutEntryType::iterator lutEntry = lutEntriesRef_.begin();
	  lutEntry != lutEntriesRef_.end(); ++lutEntry ) {
      if ( (*lutEntry)->isWithinBounds(genMuonPlusP4_afterRad.E(), genMuonMinusP4_afterRad.E()) ) {
	pRef = (*lutEntry)->getP(genMuonPlusP4_beforeRad, genMuonPlusP4_afterRad, genMuonMinusP4_beforeRad, genMuonMinusP4_afterRad);
	if ( verbosity_ ) std::cout << "pRef = " << pRef << std::endl;
	pRef_found = true;
      }
    }
    if ( !pRef_found ) {
      if ( numWarnings_ < maxWarnings_ ) {
	edm::LogWarning ("<MuonRadiationCorrWeightProducer>")
	  << "Failed to find entry in reference LUT for" 
	  << " muon+: En = " << genMuonPlusP4_afterRad.E() << ", dEn = " << (genMuonPlusP4_beforeRad.E() - genMuonPlusP4_afterRad.E()) << ";"
	  << " muon-: En = " << genMuonMinusP4_afterRad.E() << ", dEn = " << (genMuonMinusP4_beforeRad.E() - genMuonMinusP4_afterRad.E()) << " !!" << std::endl;
	++numWarnings_;
      }
      error = 1;
    }
    
    std::map<std::string, double> pOthers; // key = model name
    for ( std::map<std::string, vlutEntryType>::iterator lutEntriesOther = lutEntriesOthers_.begin();
	  lutEntriesOther != lutEntriesOthers_.end(); ++lutEntriesOther ) {
      bool pOther_found = false;
      for ( vlutEntryType::iterator lutEntry = lutEntriesOther->second.begin();
	    lutEntry != lutEntriesOther->second.end(); ++lutEntry ) {
	if ( (*lutEntry)->isWithinBounds(genMuonPlusP4_afterRad.E(), genMuonMinusP4_afterRad.E()) ) {
	  pOthers[lutEntriesOther->first] = (*lutEntry)->getP(genMuonPlusP4_beforeRad, genMuonPlusP4_afterRad, genMuonMinusP4_beforeRad, genMuonMinusP4_afterRad);
	  if ( verbosity_ ) std::cout << "pOthers[" << lutEntriesOther->first << "] = " << pOthers[lutEntriesOther->first] << std::endl;
	  pOther_found = true;
	}
      }
      if ( !pOther_found ) {
	if ( numWarnings_ < maxWarnings_ ) {
	  edm::LogWarning ("<MuonRadiationCorrWeightProducer>")
	    << "Failed to find entry in LUT = " << lutEntriesOther->first << " for"
	    << " muon+: En = " << genMuonPlusP4_afterRad.E() << ", dEn = " << (genMuonPlusP4_beforeRad.E() - genMuonPlusP4_afterRad.E()) << ";"
	    << " muon-: En = " << genMuonMinusP4_afterRad.E() << ", dEn = " << (genMuonMinusP4_beforeRad.E() - genMuonMinusP4_afterRad.E()) << " !!" << std::endl;
	  ++numWarnings_;
	}
	error = 1;
      }
    }

    if ( !error ) {
      double pMean = pRef;
      for ( std::map<std::string, double>::const_iterator pOther = pOthers.begin();
	    pOther != pOthers.end(); ++pOther ) {
	pMean += pOther->second;
      }
      pMean /= (1 + numOthers_);
      if ( verbosity_ ) std::cout << "pMean = " << pMean << std::endl;
      
      double pErr2 = square(pRef - pMean);
      for ( std::map<std::string, double>::const_iterator pOther = pOthers.begin();
	    pOther != pOthers.end(); ++pOther ) {
	pErr2 += square(pOther->second - pMean);
      }
      pErr2 /= numOthers_;
      if ( verbosity_ ) std::cout << "pErr = " << sqrt(pErr2) << std::endl;

      if ( pRef > 0. ) {
	weight = pMean/pRef;
	weightErr = sqrt(pErr2)/pRef;
      } else {
	weight = maxWeight_;
	weightErr = maxWeight_;
      }
    }
  } else {
    if ( numWarnings_ < maxWarnings_ ) {
      edm::LogWarning ("<MuonRadiationCorrWeightProducer>")
	<< "Failed to find generator level taus matching reconstructed muons !!" << std::endl;
      ++numWarnings_;
    }
  }

  if ( weight < minWeight_ ) weight = minWeight_;
  if ( weight > maxWeight_ ) weight = maxWeight_;
  double weightUp = weight + weightErr;
  if ( weightUp > (2.*maxWeight_) ) weightUp = 2.*maxWeight_;
  double weightDown = std::max(weight - weightErr, 0.);
  
  if ( verbosity_ ) {
    std::cout << "--> weight = " << weight << " + " << (weightUp - weight) << " - " << (weight - weightDown) << std::endl;
  }

  std::auto_ptr<double> weightPtr(new double(weight));
  evt.put(weightPtr, "weight");
  std::auto_ptr<double> weightUpPtr(new double(weightUp));
  evt.put(weightUpPtr, "weightUp");
  std::auto_ptr<double> weightDownPtr(new double(weightDown));
  evt.put(weightDownPtr, "weightDown");
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(MuonRadiationCorrWeightProducer);



//This file is imported from:
//Updated to: http://cmssw.cvs.cern.ch/cgi-bin/cmssw.cgi/UserCode/EGamma/EgammaCalibratedGsfElectrons/CalibratedElectronAlgos/src/ElectronEnergyCalibrator.cc?revision=1.22
//I replace here the userFloats with the variables that are now available in Electron.h?revision=1.40

//#if CMSSW_VERSION>500


#include "EgammaAnalysis/ElectronTools/interface/PatElectronEnergyCalibrator.h"


#include <CLHEP/Random/RandGaussQ.h>
#include <CLHEP/Random/Random.h>
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/Exception.h"

/****************************************************************************
 *
 * Propagate SC calibration from Zee fit to the electrons
 *
 ****************************************************************************/

using namespace edm;
/*
void ElectronEnergyCalibrator::correct
(pat::Electron & electron, const edm::Event& event , const edm::EventSetup& eventSetup ) 
  {
    correct(electron, electron.r9(), event , eventSetup, electron.ecalRegressionEnergy(), electron.ecalRegressionError());
  }
*/
void ElectronEnergyCalibrator::correct
( reco::GsfElectron & electron, double r9,  const edm::Event& event, const edm::EventSetup& eventSetup,double newEnergy, double newEnergyError) 

 {
   newEnergy_ = newEnergy;
   newEnergyError_ = newEnergyError; 

	          if (synchronization_) std::cout<< "[ElectronEnergCorrector]=============SYNCHRONIZATION MODE!================" << std::endl;

   switch (applyCorrections_){
     
	 case 0:
		//====================================================================================================
		//Do not apply the corrections
		//====================================================================================================
	          if (verbose_) std::cout<< "You choose not to apply corrections" << std::endl;
	  break;
	 case 1:
		//====================================================================================================
		//Take the REGRESSION_1
		//====================================================================================================
		//====================================================================================================
		//DEBUG BLOCK
		//====================================================================================================
                  if (verbose_) std::cout << "[ElectronEnergCorrector] R9 " << r9 << std::endl;
                  if (verbose_) std::cout << "[ElectronEnergCorrector] BEFORE comb momentum " << electron.p4(reco::GsfElectron::P4_COMBINATION).t() << std::endl;
                  if (verbose_) std::cout << "[ElectronEnergCorrector] BEFORE comb momentum error " << electron.p4Error(reco::GsfElectron::P4_COMBINATION) << std::endl;
                  if (verbose_) std::cout << "[ElectronEnergCorrector] PFlow energy " << electron.pflowSuperCluster()->rawEnergy()  << std::endl;
		//====================================================================================================
		//TAKE THE SCALE CORRECTIONS FROM SHERVIN
		//====================================================================================================
		  computeNewRegEnergy(electron, r9, event.run()) ;
		  // apply E-p combination
		  computeEpCombination(electron) ;
		  electron.correctMomentum(newMomentum_,errorTrackMomentum_,finalMomentumError_);
                  if (verbose_) std::cout << "[ElectronEnergCorrector] AFTER Regression Energy, new comb momentum " << newEnergy_ << " " << electron.p4(reco::GsfElectron::P4_COMBINATION).t() << std::endl;
	  break;
	 case 10:
	        //====================================================================================================
		//Take the REGRESSION_1 WITHOUT SCALE CORRECTIONS
		//====================================================================================================
		//====================================================================================================
		//DEBUG BLOCK
		//====================================================================================================
                  if (verbose_) std::cout << "[ElectronEnergCorrector] R9 " << r9 << std::endl;
                  if (verbose_) std::cout << "[ElectronEnergCorrector] BEFORE comb momentum " << electron.p4(reco::GsfElectron::P4_COMBINATION).t() << std::endl;
                  if (verbose_) std::cout << "[ElectronEnergCorrector] BEFORE comb momentum error " << electron.p4Error(reco::GsfElectron::P4_COMBINATION) << std::endl;
                  if (verbose_) std::cout << "[ElectronEnergCorrector] PFlow energy " << electron.pflowSuperCluster()->rawEnergy()  << std::endl;
		//====================================================================================================
		//TAKE THE SCALE CORRECTIONS FROM SHERVIN
		//====================================================================================================
		  // apply E-p combination
		  computeEpCombination(electron) ;
		  electron.correctMomentum(newMomentum_,errorTrackMomentum_,finalMomentumError_);
                  if (verbose_) std::cout << "[ElectronEnergCorrector] AFTER Regression Energy, new comb momentum " << newEnergy_ << " " << electron.p4(reco::GsfElectron::P4_COMBINATION).t() << std::endl;
	  break;
	 case 2:
		//====================================================================================================
		//Take the REGRESSION_2
		//====================================================================================================
	  break;
	 default:       
		//====================================================================================================
		//Apply the default corrections
		//====================================================================================================
		//====================================================================================================
		//DEBUG BLOCK
		//====================================================================================================
                  if (verbose_) std::cout << "[ElectronEnergCorrector] R9 " << r9 << std::endl;
                  if (verbose_) std::cout << "[ElectronEnergCorrector] BEFORE isEB, isEE, isEBEEgap " << electron.isEB() << " " <<
                          electron.isEE() << " " << electron.isEBEEGap() << std::endl;
                  if (verbose_) std::cout << "[ElectronEnergCorrector] BEFORE R9, class " << r9 << " " << 
                          electron.classification() << std::endl;
                  if (verbose_) std::cout << "[ElectronEnergCorrector] BEFORE comb momentum error " << electron.p4Error(reco::GsfElectron::P4_COMBINATION) << std::endl;
                  if (verbose_) std::cout << "[ElectronEnergCorrector] PFlow energy " << electron.pflowSuperCluster()->rawEnergy()  << std::endl;
		//====================================================================================================
		  
		  // newEnery will be overwritten in computeNewEnergy
		  newEnergy_ = electron.ecalEnergy() ;
		  newEnergyError_ = electron.ecalEnergyError(); 
		  
		  // apply ECAL calibration scale and smearing factors depending on period and categories
		  computeNewEnergy(electron, r9, event.run()) ;
		  //electron.correctEcalEnergy(newEnergy_,newEnergyError_) ;
                  
		  // apply E-p combination
		  computeEpCombination(electron) ;
		  electron.correctMomentum(newMomentum_,errorTrackMomentum_,finalMomentumError_);
                  
		//====================================================================================================
		//DEBUG BLOCK
		//====================================================================================================
                         if (verbose_) std::cout << "[ElectronEnergCorrector] AFTER ecalEnergy, new comb momentum " << newEnergy_ << " " << electron.p4(reco::GsfElectron::P4_COMBINATION).t() << std::endl;
                         if (verbose_) std::cout << "[ElectronEnergCorrector] AFTER  E/p, E/p error "<<
                           electron.eSuperClusterOverP()<<" "<<sqrt(
                       					     (newEnergyError_/electron.trackMomentumAtVtx().R())*(newEnergyError_/electron.trackMomentumAtVtx().R()) +
                       					     (newEnergy_*electron.trackMomentumError()/electron.trackMomentumAtVtx().R()/electron.trackMomentumAtVtx().R())*
                       					     (newEnergy_*electron.trackMomentumError()/electron.trackMomentumAtVtx().R()/electron.trackMomentumAtVtx().R()))<<std::endl;
                         if (verbose_) std::cout << "[ElectronEnergCorrector] AFTER comb momentum error " << electron.p4Error(reco::GsfElectron::P4_COMBINATION) << std::endl;
		//====================================================================================================


	  break;
  }
 }

void ElectronEnergyCalibrator::computeNewRegEnergy
 ( const reco::GsfElectron & electron, float r9, int run)
{
  
  
  float scale = 1.0;
  float dsigMC=0., corrMC=0.;

   edm::Service<edm::RandomNumberGenerator> rng;
   if ( ! rng.isAvailable()) {
     throw cms::Exception("Configuration")
       << "XXXXXXX requires the RandomNumberGeneratorService\n"
          "which is not present in the configuration file.  You must add the service\n"
          "in the configuration file or remove the modules that require it.";
   }
  

  if (!isMC_) {
	  if (dataset_=="2012Jul13ReReco") {                     
      // values from https://twiki.cern.ch/twiki/bin/view/CMS/ECALELF	
      if (electron.isEB() && fabs(electron.superCluster()->eta())<1 and r9<0.94) {
        if (run>=190645 && run<=190781) scale = 1.0020; 
	if (run>=190782 && run<=191042) scale = 1.0079;   
	if (run>=191043 && run<=193555) scale = 0.9989;   
	if (run>=193556 && run<=194150) scale = 0.9974;   
	if (run>=194151 && run<=194532) scale = 0.9980;   
	if (run>=194533 && run<=195113) scale = 0.9983;   
	if (run>=195114 && run<=195915) scale = 0.9984;   
	if (run>=195916 && run<=198115) scale = 0.9975;   
	if (run>=198116 && run<=199803) scale = 1.0010;   
	if (run>=199804 && run<=200048) scale = 1.0021;   
	if (run>=200049 && run<=200151) scale = 1.0035;   
	if (run>=200152 && run<=200490) scale = 1.0013;   
	if (run>=200491 && run<=200531) scale = 1.0035;   
	if (run>=200532 && run<=201656) scale = 1.0017;   
	if (run>=201657 && run<=202305) scale = 1.0026;   
	if (run>=202305 && run<=203002) scale = 1.0037;   
      } else if (electron.isEB() && fabs(electron.superCluster()->eta())<1 and r9>=0.94) {      
        if (run>=190645 && run<=190781) scale = 0.9980; 
	if (run>=190782 && run<=191042) scale = 1.0039;   
	if (run>=191043 && run<=193555) scale = 0.9949;   
	if (run>=193556 && run<=194150) scale = 0.9934;   
	if (run>=194151 && run<=194532) scale = 0.9940;   
	if (run>=194533 && run<=195113) scale = 0.9943;   
	if (run>=195114 && run<=195915) scale = 0.9944;   
	if (run>=195916 && run<=198115) scale = 0.9936;   
	if (run>=198116 && run<=199803) scale = 0.9970;   
	if (run>=199804 && run<=200048) scale = 0.9982;   
	if (run>=200049 && run<=200151) scale = 0.9996;   
	if (run>=200152 && run<=200490) scale = 0.9973;   
	if (run>=200491 && run<=200531) scale = 0.9995;   
	if (run>=200532 && run<=201656) scale = 0.9978;   
	if (run>=201657 && run<=202305) scale = 0.9986;   
	if (run>=202305 && run<=203002) scale = 0.9998;   
      } else if (electron.isEB() && fabs(electron.superCluster()->eta())>=1 and r9<0.94) {      
        if (run>=190645 && run<=190781) scale = 1.0032; 
	if (run>=190782 && run<=191042) scale = 1.0063;   
	if (run>=191043 && run<=193555) scale = 0.9998;   
	if (run>=193556 && run<=194150) scale = 0.9954;   
	if (run>=194151 && run<=194532) scale = 0.9965;   
	if (run>=194533 && run<=195113) scale = 0.9984;   
	if (run>=195114 && run<=195915) scale = 0.9977;   
	if (run>=195916 && run<=198115) scale = 0.9965;   
	if (run>=198116 && run<=199803) scale = 0.9999;   
	if (run>=199804 && run<=200048) scale = 1.0008;   
	if (run>=200049 && run<=200151) scale = 1.0017;   
	if (run>=200152 && run<=200490) scale = 1.0003;   
	if (run>=200491 && run<=200531) scale = 1.0017;   
	if (run>=200532 && run<=201656) scale = 0.9999;   
	if (run>=201657 && run<=202305) scale = 1.0003;   
	if (run>=202305 && run<=203002) scale = 1.0010;   
      } else if (electron.isEB() && fabs(electron.superCluster()->eta())>=1 and r9>=0.94) {      
        if (run>=190645 && run<=190781) scale = 0.9919; 
	if (run>=190782 && run<=191042) scale = 0.9951;   
	if (run>=191043 && run<=193555) scale = 0.9885;   
	if (run>=193556 && run<=194150) scale = 0.9841;   
	if (run>=194151 && run<=194532) scale = 0.9852;   
	if (run>=194533 && run<=195113) scale = 0.9872;   
	if (run>=195114 && run<=195915) scale = 0.9864;   
	if (run>=195916 && run<=198115) scale = 0.9852;   
	if (run>=198116 && run<=199803) scale = 0.9886;   
	if (run>=199804 && run<=200048) scale = 0.9895;   
	if (run>=200049 && run<=200151) scale = 0.9905;   
	if (run>=200152 && run<=200490) scale = 0.9890;   
	if (run>=200491 && run<=200531) scale = 0.9905;   
	if (run>=200532 && run<=201656) scale = 0.9887;   
	if (run>=201657 && run<=202305) scale = 0.9891;   
	if (run>=202305 && run<=203002) scale = 0.9897;   
      } else if (electron.isEE() && fabs(electron.superCluster()->eta())<2 and r9<0.94) {
        if (run>=190645 && run<=190781) scale = 0.9945; 
	if (run>=190782 && run<=191042) scale = 0.9996;   
	if (run>=191043 && run<=193555) scale = 0.9968;   
	if (run>=193556 && run<=194150) scale = 0.9969;   
	if (run>=194151 && run<=194532) scale = 0.9986;   
	if (run>=194533 && run<=195113) scale = 1.0006;   
	if (run>=195114 && run<=195915) scale = 1.0010;   
	if (run>=195916 && run<=198115) scale = 1.0020;   
	if (run>=198116 && run<=199803) scale = 0.9963;   
	if (run>=199804 && run<=200048) scale = 0.9965;   
	if (run>=200049 && run<=200151) scale = 0.9992;   
	if (run>=200152 && run<=200490) scale = 0.9991;   
	if (run>=200491 && run<=200531) scale = 0.9995;   
	if (run>=200532 && run<=201656) scale = 0.9978;   
	if (run>=201657 && run<=202305) scale = 0.9987;   
	if (run>=202305 && run<=203002) scale = 1.0003;   
      } else if (electron.isEE() && fabs(electron.superCluster()->eta())<2 and r9>=0.94) {      
        if (run>=190645 && run<=190781) scale = 0.9881; 
	if (run>=190782 && run<=191042) scale = 0.9932;   
	if (run>=191043 && run<=193555) scale = 0.9904;   
	if (run>=193556 && run<=194150) scale = 0.9905;   
	if (run>=194151 && run<=194532) scale = 0.9922;   
	if (run>=194533 && run<=195113) scale = 0.9943;   
	if (run>=195114 && run<=195915) scale = 0.9946;   
	if (run>=195916 && run<=198115) scale = 0.9956;   
	if (run>=198116 && run<=199803) scale = 0.9899;   
	if (run>=199804 && run<=200048) scale = 0.9901;   
	if (run>=200049 && run<=200151) scale = 0.9928;   
	if (run>=200152 && run<=200490) scale = 0.9927;   
	if (run>=200491 && run<=200531) scale = 0.9931;   
	if (run>=200532 && run<=201656) scale = 0.9914;   
	if (run>=201657 && run<=202305) scale = 0.9923;   
	if (run>=202305 && run<=203002) scale = 0.9940;   
      } else if (electron.isEE() && fabs(electron.superCluster()->eta())>=2 and r9<0.94) {      
        if (run>=190645 && run<=190781) scale = 0.9965; 
	if (run>=190782 && run<=191042) scale = 1.0010;   
	if (run>=191043 && run<=193555) scale = 0.9987;   
	if (run>=193556 && run<=194150) scale = 0.9988;   
	if (run>=194151 && run<=194532) scale = 0.9994;   
	if (run>=194533 && run<=195113) scale = 0.9999;   
	if (run>=195114 && run<=195915) scale = 1.0004;   
	if (run>=195916 && run<=198115) scale = 0.9992;   
	if (run>=198116 && run<=199803) scale = 1.0044;   
	if (run>=199804 && run<=200048) scale = 1.0060;   
	if (run>=200049 && run<=200151) scale = 1.0101;   
	if (run>=200152 && run<=200490) scale = 1.0073;   
	if (run>=200491 && run<=200531) scale = 1.0106;   
	if (run>=200532 && run<=201656) scale = 1.0069;   
	if (run>=201657 && run<=202305) scale = 1.0121;   
	if (run>=202305 && run<=203002) scale = 1.0144;   
      } else if (electron.isEE() && fabs(electron.superCluster()->eta())>=2 and r9>=0.94) {      
        if (run>=190645 && run<=190781) scale = 0.9862; 
	if (run>=190782 && run<=191042) scale = 0.9907;   
	if (run>=191043 && run<=193555) scale = 0.9884;   
	if (run>=193556 && run<=194150) scale = 0.9885;   
	if (run>=194151 && run<=194532) scale = 0.9891;   
	if (run>=194533 && run<=195113) scale = 0.9896;   
	if (run>=195114 && run<=195915) scale = 0.9900;   
	if (run>=195916 && run<=198115) scale = 0.9889;   
	if (run>=198116 && run<=199803) scale = 0.9941;   
	if (run>=199804 && run<=200048) scale = 0.9957;   
	if (run>=200049 && run<=200151) scale = 0.9999;   
	if (run>=200152 && run<=200490) scale = 0.9970;   
	if (run>=200491 && run<=200531) scale = 1.0004;   
	if (run>=200532 && run<=201656) scale = 0.9967;   
	if (run>=201657 && run<=202305) scale = 1.0018;   
	if (run>=202305 && run<=203002) scale = 1.0042;   
      }
	  }
  }
  if (dataset_=="Summer12_DR53X_HCP2012") { 
      if (electron.isEB() && fabs(electron.superCluster()->eta())<1 && r9<0.94) dsigMC = 0.0103;
      if (electron.isEB() && fabs(electron.superCluster()->eta())<1 && r9>=0.94) dsigMC = 0.0090;
      if (electron.isEB() && fabs(electron.superCluster()->eta())>=1 && r9<0.94) dsigMC = 0.0190;
      if (electron.isEB() && fabs(electron.superCluster()->eta())>=1 && r9>=0.94) dsigMC = 0.0156;
      if (electron.isEE() && fabs(electron.superCluster()->eta())<2 && r9<0.94) dsigMC = 0.0269;
      if (electron.isEE() && fabs(electron.superCluster()->eta())<2 && r9>=0.94) dsigMC = 0.0287;
      if (electron.isEE() && fabs(electron.superCluster()->eta())>=2 && r9<0.94) dsigMC = 0.0364;
      if (electron.isEE() && fabs(electron.superCluster()->eta())>=2 && r9>=0.94) dsigMC = 0.0321;   
    }

  if (!isMC_ ) newEnergy_ = newEnergy_*scale;
  if (isMC_) {
    CLHEP::RandGaussQ gaussDistribution(rng->getEngine(), 1.,dsigMC);
    corrMC = gaussDistribution.fire();
    if (verbose_) std::cout << "[ElectronEnergyCalibrator] unsmeared energy " << newEnergy_ << std::endl;
    if (synchronization_) {newEnergy_ = newEnergy_*(1+dsigMC);} else {newEnergy_ = newEnergy_*corrMC; }
    if (verbose_) std::cout << "[ElectronEnergyCalibrator] smeared energy " << newEnergy_ << std::endl;
  }  
  // correct energy error for MC and for data as error is obtained from (ideal) MC parametrisation
  if (updateEnergyError_)
   newEnergyError_ = sqrt(newEnergyError_*newEnergyError_ + dsigMC*dsigMC*newEnergy_*newEnergy_) ;
  if (verbose_) std::cout << "[ElectronEnergyCalibrator] standard ecalEnergy " << electron.ecalEnergy() << " recalibrated Regression Energy " << newEnergy_ << std::endl;
  if (verbose_) std::cout << "[ElectronEnergyCalibrator] standard ecalEnergy error " << electron.ecalEnergyError() << " recalibrated Regression Energy error " << newEnergyError_ << std::endl;

}

void ElectronEnergyCalibrator::computeNewEnergy
 ( const reco::GsfElectron & electron, float r9, int run)
 {
  //double scEnergy = electron.superCluster()->energy() ;
  double scEnergy = electron.ecalEnergy() ;
  float corr=0., scale=1.;
  float dsigMC=0., corrMC=0.;
  //  newEnergyError_ = electron.ecalEnergyError() ;

  // Compute correction depending on run, categories and dataset
  // Corrections for the PromptReco from R. Paramattti et al.
  //   https://indico.cern.ch/getFile.py/access?contribId=7&sessionId=1&resId=0&materialId=slides&confId=155805 (Oct03, PromptV6, 05Aug, 05Jul)
  //   https://indico.cern.ch/getFile.py/access?contribId=2&resId=0&materialId=slides&confId=149567 (PromptV5)
  //   https://indico.cern.ch/getFile.py/access?contribId=2&resId=0&materialId=slides&confId=149567 (05Jul)
  //   https://hypernews.cern.ch/HyperNews/CMS/get/AUX/2011/07/06/16:50:04-57776-ScaleAndResolution_20110706.pdf (May10+PromptV4)
  // Correction for the ReReco from R. paramatti et al. (private communication, AN in preparation)
  // Corrections for PromptReco are run and R9 dependant, corrections for the ReReco are categories or EB+/EB-/EE+/EE- dependant
  // Correction for MC is a gaussian smearing for the resolution, averaged from the results over the three periods
   edm::Service<edm::RandomNumberGenerator> rng;
   if ( ! rng.isAvailable()) {
     throw cms::Exception("Configuration")
       << "XXXXXXX requires the RandomNumberGeneratorService\n"
          "which is not present in the configuration file.  You must add the service\n"
          "in the configuration file or remove the modules that require it.";
   }
  
  // data corrections 
  if (!isMC_) {
    // corrections for prompt
    if (dataset_=="Prompt") {
      if (run>=160431 && run<=167784) {
	if (electron.isEB()) {
	  if (run>=160431 && run<=163869) {
            if (r9>=0.94) corr = +0.0047;
            if (r9<0.94) corr = -0.0025;
	  } else if (run>=165071 && run<=165970) {
            if (r9>=0.94) corr = +0.0007;
            if (r9<0.94) corr = -0.0049;
	  } else if (run>=165971 && run<=166502) {
            if (r9>=0.94) corr = -0.0003;
            if (r9<0.94) corr = -0.0067;
	  } else if (run>=166503 && run<=166861) {
            if (r9>=0.94) corr = -0.0011;
            if (r9<0.94) corr = -0.0063;
	  } else if (run>=166862 && run<=167784) {
            if (r9>=0.94) corr = -0.0014;
            if (r9<0.94) corr = -0.0074;
	  } 
	} else if (electron.isEE()) {
	  if (run>=160431 && run<=163869) {
            if (r9>=0.94) corr = -0.0058;
            if (r9<0.94) corr = +0.0010;
	  } else if (run>=165071 && run<=165970) {
            if (r9>=0.94) corr = -0.0249;
            if (r9<0.94) corr = -0.0062;
	  } else if (run>=165971 && run<=166502) {
            if (r9>=0.94) corr = -0.0376;
            if (r9<0.94) corr = -0.0133;
	  } else if (run>=166503 && run<=166861) {
            if (r9>=0.94) corr = -0.0450;
            if (r9<0.94) corr = -0.0178;
	  } else if (run>=166862 && run<=167784) {
            if (r9>=0.94) corr = -0.0561;
            if (r9<0.94) corr = -0.0273;
	  } 
	}    
      } else if (run>=1700053 && run <=172619) {
	if (electron.isEB()) {
	  if (r9>=0.94) corr = -0.0011;
	  if (r9<0.94) corr = -0.0067;
	} else if (electron.isEE()) {
	  if (r9>=0.94) corr = +0.0009;
	  if (r9<0.94) corr = -0.0046;
	}  
      } else if (run>=172620 && run <=175770) {
	if (electron.isEB()) {
	  if (r9>=0.94) corr = -0.0046;
	  if (r9<0.94) corr = -0.0104;
	} else if (electron.isEE()) {
	  if (r9>=0.94) corr = +0.0337;
	  if (r9<0.94) corr = +0.0250;
        }  
      } else if (run>=175860 && run<=177139) {                      // prompt-v1 corrections for 2011B [ 175860 - 177139 ]
        if (electron.isEB() && fabs(electron.superCluster()->eta())>=1 and r9<0.94) corr = -0.0228;
        if (electron.isEB() && fabs(electron.superCluster()->eta())>=1 and r9>=0.94) corr = -0.0118;
        if (electron.isEB() && fabs(electron.superCluster()->eta())<1 and r9<0.94) corr = -0.0075;
        if (electron.isEB() && fabs(electron.superCluster()->eta())<1 and r9>=0.94) corr = -0.0034;
        if (electron.isEE() && fabs(electron.superCluster()->eta())>=2 and r9<0.94) corr = -0.0041;
        if (electron.isEE() && fabs(electron.superCluster()->eta())>=2 and r9>=0.94) corr = +0.0019;
        if (electron.isEE() && fabs(electron.superCluster()->eta())<2 and r9<0.94) corr = +0.0147;
        if (electron.isEE() && fabs(electron.superCluster()->eta())<2 and r9>=0.94) corr = +0.0168;
      } else if (run>=177140 && run<=178421) {                      // prompt-v1 corrections for 2011B [ 177140 - 178421 ]
        if (electron.isEB() && fabs(electron.superCluster()->eta())>=1 and r9<0.94) corr = -0.0239;
        if (electron.isEB() && fabs(electron.superCluster()->eta())>=1 and r9>=0.94) corr = -0.0129;
        if (electron.isEB() && fabs(electron.superCluster()->eta())<1 and r9<0.94) corr = -0.0079;
        if (electron.isEB() && fabs(electron.superCluster()->eta())<1 and r9>=0.94) corr = -0.0038;
        if (electron.isEE() && fabs(electron.superCluster()->eta())>=2 and r9<0.94) corr = -0.0011;
        if (electron.isEE() && fabs(electron.superCluster()->eta())>=2 and r9>=0.94) corr = +0.0049;
        if (electron.isEE() && fabs(electron.superCluster()->eta())<2 and r9<0.94) corr = +0.0236;
        if (electron.isEE() && fabs(electron.superCluster()->eta())<2 and r9>=0.94) corr = +0.0257;
      } else if (run>=178424 && run<=180252) {                      // prompt-v1 corrections for 2011B [ 178424 - 180252 ]
        if (electron.isEB() && fabs(electron.superCluster()->eta())>=1 and r9<0.94) corr = -0.0260;
        if (electron.isEB() && fabs(electron.superCluster()->eta())>=1 and r9>=0.94) corr = -0.0150;
        if (electron.isEB() && fabs(electron.superCluster()->eta())<1 and r9<0.94) corr = -0.0094;
        if (electron.isEB() && fabs(electron.superCluster()->eta())<1 and r9>=0.94) corr = -0.0052;
        if (electron.isEE() && fabs(electron.superCluster()->eta())>=2 and r9<0.94) corr = -0.0050;
        if (electron.isEE() && fabs(electron.superCluster()->eta())>=2 and r9>=0.94) corr = +0.0009;
        if (electron.isEE() && fabs(electron.superCluster()->eta())<2 and r9<0.94) corr = +0.0331;
        if (electron.isEE() && fabs(electron.superCluster()->eta())<2 and r9>=0.94) corr = +0.0353;
      } 
    // corrections for rereco  
    } else if (dataset_=="ReReco") {                     // corrections for ReReco
      // values from https://indico.cern.ch/conferenceDisplay.py?confId=146386
      if (run>=160329 && run <=168437) {                 // Jul05 period 160329-168437
        if (electron.isEB() && fabs(electron.superCluster()->eta())>=1 and r9<0.94) corr = -0.0150;
        if (electron.isEB() && fabs(electron.superCluster()->eta())>=1 and r9>=0.94) corr = -0.0039;
        if (electron.isEB() && fabs(electron.superCluster()->eta())<1 and r9<0.94) corr = -0.0014;
        if (electron.isEB() && fabs(electron.superCluster()->eta())<1 and r9>=0.94) corr = +0.0028;
        if (electron.isEE() && fabs(electron.superCluster()->eta())>=2 and r9<0.94) corr = -0.0050;
        if (electron.isEE() && fabs(electron.superCluster()->eta())>=2 and r9>=0.94) corr = +0.0010;
        if (electron.isEE() && fabs(electron.superCluster()->eta())<2 and r9<0.94) corr = -0.0025;
        if (electron.isEE() && fabs(electron.superCluster()->eta())<2 and r9>=0.94) corr = -0.0005;
      } else if (run>=170053 && run <=172619) {          // Aug05 period 170053-172619
        if (electron.isEB() && fabs(electron.superCluster()->eta())>=1 and r9<0.94) corr = -0.0191;
        if (electron.isEB() && fabs(electron.superCluster()->eta())>=1 and r9>=0.94) corr = -0.0081;
        if (electron.isEB() && fabs(electron.superCluster()->eta())<1 and r9<0.94) corr = -0.0030;
        if (electron.isEB() && fabs(electron.superCluster()->eta())<1 and r9>=0.94) corr = +0.0012;
        if (electron.isEE() && fabs(electron.superCluster()->eta())>=2 and r9<0.94) corr = +0.0052;
        if (electron.isEE() && fabs(electron.superCluster()->eta())>=2 and r9>=0.94) corr = +0.0113;
        if (electron.isEE() && fabs(electron.superCluster()->eta())<2 and r9<0.94) corr = +0.0041;
        if (electron.isEE() && fabs(electron.superCluster()->eta())<2 and r9>=0.94) corr = +0.0062;
      } else if (run>=172620 && run <=175770) {          // Oct03 period
        if (electron.isEB() && fabs(electron.superCluster()->eta())>=1 and r9<0.94) corr = -0.0150;
        if (electron.isEB() && fabs(electron.superCluster()->eta())>=1 and r9>=0.94) corr = -0.0039;
        if (electron.isEB() && fabs(electron.superCluster()->eta())<1 and r9<0.94) corr = +0.0001;
        if (electron.isEB() && fabs(electron.superCluster()->eta())<1 and r9>=0.94) corr = +0.0043;
        if (electron.isEE() && fabs(electron.superCluster()->eta())>=2 and r9<0.94) corr = +0.0001;
        if (electron.isEE() && fabs(electron.superCluster()->eta())>=2 and r9>=0.94) corr = +0.0062;
        if (electron.isEE() && fabs(electron.superCluster()->eta())<2 and r9<0.94) corr = +0.0026;
        if (electron.isEE() && fabs(electron.superCluster()->eta())<2 and r9>=0.94) corr = +0.0048;
      } else if (run>=175860 && run<=177139) {                      // prompt-v1 corrections for 2011B [ 175860 - 177139 ]
        if (electron.isEB() && fabs(electron.superCluster()->eta())>=1 and r9<0.94) corr = -0.0228;
        if (electron.isEB() && fabs(electron.superCluster()->eta())>=1 and r9>=0.94) corr = -0.0118;
        if (electron.isEB() && fabs(electron.superCluster()->eta())<1 and r9<0.94) corr = -0.0075;
        if (electron.isEB() && fabs(electron.superCluster()->eta())<1 and r9>=0.94) corr = -0.0034;
        if (electron.isEE() && fabs(electron.superCluster()->eta())>=2 and r9<0.94) corr = -0.0041;
        if (electron.isEE() && fabs(electron.superCluster()->eta())>=2 and r9>=0.94) corr = +0.0019;
        if (electron.isEE() && fabs(electron.superCluster()->eta())<2 and r9<0.94) corr = +0.0147;
        if (electron.isEE() && fabs(electron.superCluster()->eta())<2 and r9>=0.94) corr = +0.0168;
      } else if (run>=177140 && run<=178421) {                      // prompt-v1 corrections for 2011B [ 177140 - 178421 ]
        if (electron.isEB() && fabs(electron.superCluster()->eta())>=1 and r9<0.94) corr = -0.0239;
        if (electron.isEB() && fabs(electron.superCluster()->eta())>=1 and r9>=0.94) corr = -0.0129;
        if (electron.isEB() && fabs(electron.superCluster()->eta())<1 and r9<0.94) corr = -0.0079;
        if (electron.isEB() && fabs(electron.superCluster()->eta())<1 and r9>=0.94) corr = -0.0038;
        if (electron.isEE() && fabs(electron.superCluster()->eta())>=2 and r9<0.94) corr = -0.0011;
        if (electron.isEE() && fabs(electron.superCluster()->eta())>=2 and r9>=0.94) corr = +0.0049;
        if (electron.isEE() && fabs(electron.superCluster()->eta())<2 and r9<0.94) corr = +0.0236;
        if (electron.isEE() && fabs(electron.superCluster()->eta())<2 and r9>=0.94) corr = +0.0257;
      } else if (run>=178424 && run<=180252) {                      // prompt-v1 corrections for 2011B [ 178424 - 180252 ]
        if (electron.isEB() && fabs(electron.superCluster()->eta())>=1 and r9<0.94) corr = -0.0260;
        if (electron.isEB() && fabs(electron.superCluster()->eta())>=1 and r9>=0.94) corr = -0.0150;
        if (electron.isEB() && fabs(electron.superCluster()->eta())<1 and r9<0.94) corr = -0.0094;
        if (electron.isEB() && fabs(electron.superCluster()->eta())<1 and r9>=0.94) corr = -0.0052;
        if (electron.isEE() && fabs(electron.superCluster()->eta())>=2 and r9<0.94) corr = -0.0050;
        if (electron.isEE() && fabs(electron.superCluster()->eta())>=2 and r9>=0.94) corr = +0.0009;
        if (electron.isEE() && fabs(electron.superCluster()->eta())<2 and r9<0.94) corr = +0.0331;
        if (electron.isEE() && fabs(electron.superCluster()->eta())<2 and r9>=0.94) corr = +0.0353;
      } 
    // corrections for januray 16 rereco  
    } else if (dataset_=="Jan16ReReco") {                     // corrections for january 16 ReReco
      // values from http://indico.cern.ch/getFile.py/access?contribId=2&resId=0&materialId=slides&confId=176520
      if (electron.isEB() && fabs(electron.superCluster()->eta())<1 and r9<0.94) {
        if (run>=160431 && run<=167913) corr = -0.0014; 
	if (run>=170000 && run<=172619) corr = -0.0016;   
	if (run>=172620 && run<=173692) corr = -0.0017;  
	if (run>=175830 && run<=177139) corr = -0.0021;
	if (run>=177140 && run<=178421) corr = -0.0025;
	if (run>=178424 && run<=180252) corr = -0.0024;
      } else if (electron.isEB() && fabs(electron.superCluster()->eta())<1 and r9>=0.94) {      
        if (run>=160431 && run<=167913) corr = 0.0059; 
	if (run>=170000 && run<=172619) corr = 0.0046;   
	if (run>=172620 && run<=173692) corr = 0.0045;  
	if (run>=175830 && run<=177139) corr = 0.0042;
	if (run>=177140 && run<=178421) corr = 0.0038;
	if (run>=178424 && run<=180252) corr = 0.0039;
      } else if (electron.isEB() && fabs(electron.superCluster()->eta())>=1 and r9<0.94) {      
        if (run>=160431 && run<=167913) corr = -0.0045; 
	if (run>=170000 && run<=172619) corr = -0.0066;   
	if (run>=172620 && run<=173692) corr = -0.0058;  
	if (run>=175830 && run<=177139) corr = -0.0073;
	if (run>=177140 && run<=178421) corr = -0.0075;
	if (run>=178424 && run<=180252) corr = -0.0071;
      } else if (electron.isEB() && fabs(electron.superCluster()->eta())>=1 and r9>=0.94) {      
        if (run>=160431 && run<=167913) corr = 0.0084; 
	if (run>=170000 && run<=172619) corr = 0.0063;   
	if (run>=172620 && run<=173692) corr = 0.0071;  
	if (run>=175830 && run<=177139) corr = 0.0056;
	if (run>=177140 && run<=178421) corr = 0.0054;
	if (run>=178424 && run<=180252) corr = 0.0058;
      } else if (electron.isEE() && fabs(electron.superCluster()->eta())<2 and r9<0.94) {
        if (run>=160431 && run<=167913) corr = -0.0082; 
	if (run>=170000 && run<=172619) corr = -0.0025;   
	if (run>=172620 && run<=173692) corr = -0.0035;  
	if (run>=175830 && run<=177139) corr = -0.0017;
	if (run>=177140 && run<=178421) corr = -0.0010;
	if (run>=178424 && run<=180252) corr = 0.0030;
      } else if (electron.isEE() && fabs(electron.superCluster()->eta())<2 and r9>=0.94) {      
        if (run>=160431 && run<=167913) corr = -0.0033; 
	if (run>=170000 && run<=172619) corr = 0.0024;   
	if (run>=172620 && run<=173692) corr = 0.0014;  
	if (run>=175830 && run<=177139) corr = 0.0032;
	if (run>=177140 && run<=178421) corr = 0.0040;
	if (run>=178424 && run<=180252) corr = 0.0079;
      } else if (electron.isEE() && fabs(electron.superCluster()->eta())>=2 and r9<0.94) {      
        if (run>=160431 && run<=167913) corr = -0.0064; 
	if (run>=170000 && run<=172619) corr = -0.0046;   
	if (run>=172620 && run<=173692) corr = -0.0029;  
	if (run>=175830 && run<=177139) corr = -0.0040;
	if (run>=177140 && run<=178421) corr = -0.0050;
	if (run>=178424 && run<=180252) corr = -0.0059;
      } else if (electron.isEE() && fabs(electron.superCluster()->eta())>=2 and r9>=0.94) {      
        if (run>=160431 && run<=167913) corr = 0.0042; 
	if (run>=170000 && run<=172619) corr = 0.0060;   
	if (run>=172620 && run<=173692) corr = 0.0077;  
	if (run>=175830 && run<=177139) corr = 0.0067;
	if (run>=177140 && run<=178421) corr = 0.0056;
	if (run>=178424 && run<=180252) corr = 0.0047;
      } 
    // corrections for2012A and 2012B 
    } else if (dataset_=="ICHEP2012") {                     
      // values from https://hypernews.cern.ch/HyperNews/CMS/get/higgs2g/873.html	
      if (electron.isEB() && fabs(electron.superCluster()->eta())<1 and r9<0.94) {
        if (run>=190450 && run<=190781) scale = 1.0021; 
	if (run>=190782 && run<=190949) scale = 1.0154;   
	if (run>=190950 && run<=191833) scale = 1.0046;   
	if (run>=191834 && run<=193686) scale = 1.0017;   
	if (run>=193746 && run<=194210) scale = 1.0020;   
	if (run>=194211 && run<=194479) scale = 1.0037;   
	if (run>=194480 && run<=195147) scale = 1.0047;   
	if (run>=195148 && run<=195350) scale = 1.0053;   
	if (run>=195396 && run<=195530) scale = 1.0042;   
	if (run>=195531 && run<=196531) scale = 0.9981;   
      } else if (electron.isEB() && fabs(electron.superCluster()->eta())<1 and r9>=0.94) {      
        if (run>=190450 && run<=190781) scale = 0.9962; 
	if (run>=190782 && run<=190949) scale = 1.0096;   
	if (run>=190950 && run<=191833) scale = 0.9988;   
	if (run>=191834 && run<=193686) scale = 0.9958;   
	if (run>=193746 && run<=194210) scale = 0.9962;   
	if (run>=194211 && run<=194479) scale = 0.9979;   
	if (run>=194480 && run<=195147) scale = 0.9989;   
	if (run>=195148 && run<=195350) scale = 0.9995;   
	if (run>=195396 && run<=195530) scale = 0.9984;   
	if (run>=195531 && run<=196531) scale = 0.9922;   
      } else if (electron.isEB() && fabs(electron.superCluster()->eta())>=1 and r9<0.94) {      
        if (run>=190450 && run<=190781) scale = 1.0133; 
	if (run>=190782 && run<=190949) scale = 0.9997;   
	if (run>=190950 && run<=191833) scale = 1.0134;   
	if (run>=191834 && run<=193686) scale = 1.0104;   
	if (run>=193746 && run<=194210) scale = 1.0094;   
	if (run>=194211 && run<=194479) scale = 1.0118;   
	if (run>=194480 && run<=195147) scale = 1.0137;   
	if (run>=195148 && run<=195350) scale = 1.0142;   
	if (run>=195396 && run<=195530) scale = 1.0129;   
	if (run>=195531 && run<=196531) scale = 1.0065;   
      } else if (electron.isEB() && fabs(electron.superCluster()->eta())>=1 and r9>=0.94) {      
        if (run>=190450 && run<=190781) scale = 1.0020; 
	if (run>=190782 && run<=190949) scale = 0.9883;   
	if (run>=190950 && run<=191833) scale = 1.0021;   
	if (run>=191834 && run<=193686) scale = 0.9991;   
	if (run>=193746 && run<=194210) scale = 0.9980;   
	if (run>=194211 && run<=194479) scale = 1.0005;   
	if (run>=194480 && run<=195147) scale = 1.0024;   
	if (run>=195148 && run<=195350) scale = 1.0029;   
	if (run>=195396 && run<=195530) scale = 1.0016;   
	if (run>=195531 && run<=196531) scale = 0.9951;   
      } else if (electron.isEE() && fabs(electron.superCluster()->eta())<2 and r9<0.94) {
        if (run>=190450 && run<=190781) scale = 0.9989; 
	if (run>=190782 && run<=190949) scale = 1.0123;   
	if (run>=190950 && run<=191833) scale = 1.0042;   
	if (run>=191834 && run<=193686) scale = 1.0037;   
	if (run>=193746 && run<=194210) scale = 1.0047;   
	if (run>=194211 && run<=194479) scale = 1.0037;   
	if (run>=194480 && run<=195147) scale = 1.0030;   
	if (run>=195148 && run<=195350) scale = 1.0051;   
	if (run>=195396 && run<=195530) scale = 1.0003;   
	if (run>=195531 && run<=196531) scale = 1.0052;   
      } else if (electron.isEE() && fabs(electron.superCluster()->eta())<2 and r9>=0.94) {      
        if (run>=190450 && run<=190781) scale = 0.9931; 
	if (run>=190782 && run<=190949) scale = 1.0066;   
	if (run>=190950 && run<=191833) scale = 0.9985;   
	if (run>=191834 && run<=193686) scale = 0.9979;   
	if (run>=193746 && run<=194210) scale = 0.9990;   
	if (run>=194211 && run<=194479) scale = 0.9979;   
	if (run>=194480 && run<=195147) scale = 0.9972;   
	if (run>=195148 && run<=195350) scale = 0.9994;   
	if (run>=195396 && run<=195530) scale = 0.9945;   
	if (run>=195531 && run<=196531) scale = 0.9994;   
      } else if (electron.isEE() && fabs(electron.superCluster()->eta())>=2 and r9<0.94) {      
        if (run>=190450 && run<=190781) scale = 1.0052; 
	if (run>=190782 && run<=190949) scale = 1.0077;   
	if (run>=190950 && run<=191833) scale = 0.9900;   
	if (run>=191834 && run<=193686) scale = 0.9893;   
	if (run>=193746 && run<=194210) scale = 1.0042;   
	if (run>=194211 && run<=194479) scale = 1.0036;   
	if (run>=194480 && run<=195147) scale = 1.0069;   
	if (run>=195148 && run<=195350) scale = 1.0133;   
	if (run>=195396 && run<=195530) scale = 0.9999;   
	if (run>=195531 && run<=196531) scale = 1.0475;   
      } else if (electron.isEE() && fabs(electron.superCluster()->eta())>=2 and r9>=0.94) {      
        if (run>=190450 && run<=190781) scale = 0.9899; 
	if (run>=190782 && run<=190949) scale = 0.9924;   
	if (run>=190950 && run<=191833) scale = 0.9745;   
	if (run>=191834 && run<=193686) scale = 0.9738;   
	if (run>=193746 && run<=194210) scale = 0.9889;   
	if (run>=194211 && run<=194479) scale = 0.9883;   
	if (run>=194480 && run<=195147) scale = 0.9916;   
	if (run>=195148 && run<=195350) scale = 0.9982;   
	if (run>=195396 && run<=195530) scale = 0.9845;   
	if (run>=195531 && run<=196531) scale = 1.0329;   
      } 
    }else if (dataset_=="2012Jul13ReReco") {                     
      // values from https://twiki.cern.ch/twiki/bin/view/CMS/ECALELF	
      if (electron.isEB() && fabs(electron.superCluster()->eta())<1 and r9<0.94) {
        if (run>=190645 && run<=190781) scale = 1.0057; 
	if (run>=190782 && run<=191042) scale = 1.0115;   
	if (run>=191043 && run<=193555) scale = 1.0029;   
	if (run>=193556 && run<=194150) scale = 1.0018;   
	if (run>=194151 && run<=194532) scale = 1.0016;   
	if (run>=194533 && run<=195113) scale = 1.0015;   
	if (run>=195114 && run<=195915) scale = 1.0015;   
	if (run>=195916 && run<=198115) scale = 1.0006;   
	if (run>=198116 && run<=199803) scale = 1.0046;   
	if (run>=199804 && run<=200048) scale = 1.0053;   
	if (run>=200049 && run<=200151) scale = 1.0064;   
	if (run>=200152 && run<=200490) scale = 1.0045;   
	if (run>=200491 && run<=200531) scale = 1.0058;   
	if (run>=200532 && run<=201656) scale = 1.0045;   
	if (run>=201657 && run<=202305) scale = 1.0053;   
	if (run>=202305 && run<=203002) scale = 1.0065;   
      } else if (electron.isEB() && fabs(electron.superCluster()->eta())<1 and r9>=0.94) {      
        if (run>=190645 && run<=190781) scale = 1.0042; 
	if (run>=190782 && run<=191042) scale = 1.0099;   
	if (run>=191043 && run<=193555) scale = 1.0014;   
	if (run>=193556 && run<=194150) scale = 1.0002;   
	if (run>=194151 && run<=194532) scale = 1.0001;   
	if (run>=194533 && run<=195113) scale = 1.0000;   
	if (run>=195114 && run<=195915) scale = 0.9999;   
	if (run>=195916 && run<=198115) scale = 0.9991;   
	if (run>=198116 && run<=199803) scale = 1.0031;   
	if (run>=199804 && run<=200048) scale = 1.0037;   
	if (run>=200049 && run<=200151) scale = 1.0049;   
	if (run>=200152 && run<=200490) scale = 1.0029;   
	if (run>=200491 && run<=200531) scale = 1.0042;   
	if (run>=200532 && run<=201656) scale = 1.0030;   
	if (run>=201657 && run<=202305) scale = 1.0038;   
	if (run>=202305 && run<=203002) scale = 1.0050;   
      } else if (electron.isEB() && fabs(electron.superCluster()->eta())>=1 and r9<0.94) {      
        if (run>=190645 && run<=190781) scale = 1.0048; 
	if (run>=190782 && run<=191042) scale = 1.0070;   
	if (run>=191043 && run<=193555) scale = 1.0017;   
	if (run>=193556 && run<=194150) scale = 0.9979;   
	if (run>=194151 && run<=194532) scale = 0.9980;   
	if (run>=194533 && run<=195113) scale = 0.9993;   
	if (run>=195114 && run<=195915) scale = 0.9981;   
	if (run>=195916 && run<=198115) scale = 0.9971;   
	if (run>=198116 && run<=199803) scale = 1.0014;   
	if (run>=199804 && run<=200048) scale = 1.0020;   
	if (run>=200049 && run<=200151) scale = 1.0026;   
	if (run>=200152 && run<=200490) scale = 1.0012;   
	if (run>=200491 && run<=200531) scale = 1.0014;   
	if (run>=200532 && run<=201656) scale = 1.0002;   
	if (run>=201657 && run<=202305) scale = 1.0002;   
	if (run>=202305 && run<=203002) scale = 1.0010;   
      } else if (electron.isEB() && fabs(electron.superCluster()->eta())>=1 and r9>=0.94) {      
        if (run>=190645 && run<=190781) scale = 0.9953; 
	if (run>=190782 && run<=191042) scale = 0.9975;   
	if (run>=191043 && run<=193555) scale = 0.9921;   
	if (run>=193556 && run<=194150) scale = 0.9883;   
	if (run>=194151 && run<=194532) scale = 0.9884;   
	if (run>=194533 && run<=195113) scale = 0.9897;   
	if (run>=195114 && run<=195915) scale = 0.9884;   
	if (run>=195916 && run<=198115) scale = 0.9875;   
	if (run>=198116 && run<=199803) scale = 0.9918;   
	if (run>=199804 && run<=200048) scale = 0.9924;   
	if (run>=200049 && run<=200151) scale = 0.9930;   
	if (run>=200152 && run<=200490) scale = 0.9916;   
	if (run>=200491 && run<=200531) scale = 0.9918;   
	if (run>=200532 && run<=201656) scale = 0.9906;   
	if (run>=201657 && run<=202305) scale = 0.9906;   
	if (run>=202305 && run<=203002) scale = 0.9914;   
      } else if (electron.isEE() && fabs(electron.superCluster()->eta())<2 and r9<0.94) {
        if (run>=190645 && run<=190781) scale = 0.9953; 
	if (run>=190782 && run<=191042) scale = 1.0006;   
	if (run>=191043 && run<=193555) scale = 0.9999;   
	if (run>=193556 && run<=194150) scale = 1.0009;   
	if (run>=194151 && run<=194532) scale = 1.0019;   
	if (run>=194533 && run<=195113) scale = 1.0018;   
	if (run>=195114 && run<=195915) scale = 1.0019;   
	if (run>=195916 && run<=198115) scale = 1.0035;   
	if (run>=198116 && run<=199803) scale = 0.9982;   
	if (run>=199804 && run<=200048) scale = 0.9972;   
	if (run>=200049 && run<=200151) scale = 1.0001;   
	if (run>=200152 && run<=200490) scale = 1.0002;   
	if (run>=200491 && run<=200531) scale = 0.9971;   
	if (run>=200532 && run<=201656) scale = 0.9978;   
	if (run>=201657 && run<=202305) scale = 0.9984;   
	if (run>=202305 && run<=203002) scale = 1.0003;   
      } else if (electron.isEE() && fabs(electron.superCluster()->eta())<2 and r9>=0.94) {      
        if (run>=190645 && run<=190781) scale = 0.9878; 
	if (run>=190782 && run<=191042) scale = 0.9931;   
	if (run>=191043 && run<=193555) scale = 0.9924;   
	if (run>=193556 && run<=194150) scale = 0.9934;   
	if (run>=194151 && run<=194532) scale = 0.9945;   
	if (run>=194533 && run<=195113) scale = 0.9943;   
	if (run>=195114 && run<=195915) scale = 0.9945;   
	if (run>=195916 && run<=198115) scale = 0.9961;   
	if (run>=198116 && run<=199803) scale = 0.9907;   
	if (run>=199804 && run<=200048) scale = 0.9898;   
	if (run>=200049 && run<=200151) scale = 0.9927;   
	if (run>=200152 && run<=200490) scale = 0.9928;   
	if (run>=200491 && run<=200531) scale = 0.9896;   
	if (run>=200532 && run<=201656) scale = 0.9903;   
	if (run>=201657 && run<=202305) scale = 0.9910;   
	if (run>=202305 && run<=203002) scale = 0.9928;   
      } else if (electron.isEE() && fabs(electron.superCluster()->eta())>=2 and r9<0.94) {      
        if (run>=190645 && run<=190781) scale = 1.0032; 
	if (run>=190782 && run<=191042) scale = 1.0085;   
	if (run>=191043 && run<=193555) scale = 1.0074;   
	if (run>=193556 && run<=194150) scale = 1.0082;   
	if (run>=194151 && run<=194532) scale = 1.0086;   
	if (run>=194533 && run<=195113) scale = 1.0069;   
	if (run>=195114 && run<=195915) scale = 1.0070;   
	if (run>=195916 && run<=198115) scale = 1.0066;   
	if (run>=198116 && run<=199803) scale = 1.0127;   
	if (run>=199804 && run<=200048) scale = 1.0132;   
	if (run>=200049 && run<=200151) scale = 1.0161;   
	if (run>=200152 && run<=200490) scale = 1.0155;   
	if (run>=200491 && run<=200531) scale = 1.0161;   
	if (run>=200532 && run<=201656) scale = 1.0138;   
	if (run>=201657 && run<=202305) scale = 1.0182;   
	if (run>=202305 && run<=203002) scale = 1.0210;   
      } else if (electron.isEE() && fabs(electron.superCluster()->eta())>=2 and r9>=0.94) {      
        if (run>=190645 && run<=190781) scale = 0.9897; 
	if (run>=190782 && run<=191042) scale = 0.9951;   
	if (run>=191043 && run<=193555) scale = 0.9939;   
	if (run>=193556 && run<=194150) scale = 0.9947;   
	if (run>=194151 && run<=194532) scale = 0.9952;   
	if (run>=194533 && run<=195113) scale = 0.9935;   
	if (run>=195114 && run<=195915) scale = 0.9935;   
	if (run>=195916 && run<=198115) scale = 0.9931;   
	if (run>=198116 && run<=199803) scale = 0.9993;   
	if (run>=199804 && run<=200048) scale = 0.9998;   
	if (run>=200049 && run<=200151) scale = 1.0028;   
	if (run>=200152 && run<=200490) scale = 1.0021;   
	if (run>=200491 && run<=200531) scale = 1.0028;   
	if (run>=200532 && run<=201656) scale = 1.0004;   
	if (run>=201657 && run<=202305) scale = 1.0049;   
	if (run>=202305 && run<=203002) scale = 1.0077;   
      } 
    }     
  } 
    // MC smearing dsig is needed also for data for theenergy error, take it from the last MC values consistant
    // with the data choice
//  else  { // MC corrections
    if (dataset_=="Summer11"||dataset_=="ReReco") { // values from https://indico.cern.ch/conferenceDisplay.py?confId=146386
      if (electron.isEB() && fabs(electron.superCluster()->eta())<1 && r9<0.94) dsigMC = 0.01;
      if (electron.isEB() && fabs(electron.superCluster()->eta())<1 && r9>=0.94) dsigMC = 0.0099;
      if (electron.isEB() && fabs(electron.superCluster()->eta())>=1 && r9<0.94) dsigMC = 0.0217;
      if (electron.isEB() && fabs(electron.superCluster()->eta())>=1 && r9>=0.94) dsigMC = 0.0157;
      if (electron.isEE() && fabs(electron.superCluster()->eta())<2 && r9<0.94) dsigMC = 0.0326;
      if (electron.isEE() && fabs(electron.superCluster()->eta())<2 && r9>=0.94) dsigMC = 0.0330;
      if (electron.isEE() && fabs(electron.superCluster()->eta())>=2 && r9<0.94) dsigMC = 0.0331;
      if (electron.isEE() && fabs(electron.superCluster()->eta())>=2 && r9>=0.94) dsigMC = 0.0378;
    } else if (dataset_=="Fall11"||dataset_=="Jan16ReReco") { // values from https://hypernews.cern.ch/HyperNews/CMS/get/higgs2g/634.html, consistant with Jan16ReReco corrections
      if (electron.isEB() && fabs(electron.superCluster()->eta())<1 && r9<0.94) dsigMC = 0.0096;
      if (electron.isEB() && fabs(electron.superCluster()->eta())<1 && r9>=0.94) dsigMC = 0.0074;
      if (electron.isEB() && fabs(electron.superCluster()->eta())>=1 && r9<0.94) dsigMC = 0.0196;
      if (electron.isEB() && fabs(electron.superCluster()->eta())>=1 && r9>=0.94) dsigMC = 0.0141;
      if (electron.isEE() && fabs(electron.superCluster()->eta())<2 && r9<0.94) dsigMC = 0.0279;
      if (electron.isEE() && fabs(electron.superCluster()->eta())<2 && r9>=0.94) dsigMC = 0.0268;
      if (electron.isEE() && fabs(electron.superCluster()->eta())>=2 && r9<0.94) dsigMC = 0.0301;
      if (electron.isEE() && fabs(electron.superCluster()->eta())>=2 && r9>=0.94) dsigMC = 0.0293;   
    } else if (dataset_=="Summer12"||dataset_=="ICHEP2012") { 
      // new values from https://twiki.cern.ch/twiki/pub/CMS/EcalEnergyResolutionWithZee/oriented-ICHEP-scales_resolution.pdf
      if (electron.isEB() && fabs(electron.superCluster()->eta())<1 && r9<0.94) dsigMC = 0.0119;
      if (electron.isEB() && fabs(electron.superCluster()->eta())<1 && r9>=0.94) dsigMC = 0.0107;
      if (electron.isEB() && fabs(electron.superCluster()->eta())>=1 && r9<0.94) dsigMC = 0.0240;
      if (electron.isEB() && fabs(electron.superCluster()->eta())>=1 && r9>=0.94) dsigMC = 0.0149;
      if (electron.isEE() && fabs(electron.superCluster()->eta())<2 && r9<0.94) dsigMC = 0.0330;
      if (electron.isEE() && fabs(electron.superCluster()->eta())<2 && r9>=0.94) dsigMC = 0.0375;
      if (electron.isEE() && fabs(electron.superCluster()->eta())>=2 && r9<0.94) dsigMC = 0.0602;
      if (electron.isEE() && fabs(electron.superCluster()->eta())>=2 && r9>=0.94) dsigMC = 0.0607;   
    }  else if (dataset_=="Summer12_DR53X_HCP2012") { 
      if (electron.isEB() && fabs(electron.superCluster()->eta())<1 && r9<0.94) dsigMC = 0.0099;
      if (electron.isEB() && fabs(electron.superCluster()->eta())<1 && r9>=0.94) dsigMC = 0.0103;
      if (electron.isEB() && fabs(electron.superCluster()->eta())>=1 && r9<0.94) dsigMC = 0.0219;
      if (electron.isEB() && fabs(electron.superCluster()->eta())>=1 && r9>=0.94) dsigMC = 0.0158;
      if (electron.isEE() && fabs(electron.superCluster()->eta())<2 && r9<0.94) dsigMC = 0.0222;
      if (electron.isEE() && fabs(electron.superCluster()->eta())<2 && r9>=0.94) dsigMC = 0.0298;
      if (electron.isEE() && fabs(electron.superCluster()->eta())>=2 && r9<0.94) dsigMC = 0.0318;
      if (electron.isEE() && fabs(electron.superCluster()->eta())>=2 && r9>=0.94) dsigMC = 0.0302;   
    }

  
  // now correct the energy
  // intial corrections based on deltaP
  if (!isMC_ && corr!=0.) newEnergy_ = scEnergy/(1+corr);
  // new format
  if (!isMC_ && corr==0.) newEnergy_ = scEnergy*scale;
  // smearing for MC
  if (isMC_) {
    CLHEP::RandGaussQ gaussDistribution(rng->getEngine(), 1.,dsigMC);
    corrMC = gaussDistribution.fire();
    if (verbose_) std::cout << "[ElectronEnergyCalibrator] unsmeared energy " << scEnergy << std::endl;
    if (synchronization_) {newEnergy_ = newEnergy_*(1+dsigMC);} else {newEnergy_ = newEnergy_*corrMC; }
    if (verbose_) std::cout << "[ElectronEnergyCalibrator] smeared energy " << newEnergy_ << std::endl;
  }  
  // correct energy error for MC and for data as error is obtained from (ideal) MC parametrisation
  if (updateEnergyError_)
   newEnergyError_ = sqrt(newEnergyError_*newEnergyError_ + dsigMC*dsigMC*newEnergy_*newEnergy_) ;
  if (verbose_) std::cout << "[ElectronEnergyCalibrator] ecalEnergy " << electron.ecalEnergy() << " recalibrated ecalEnergy " << newEnergy_ << std::endl;
  if (verbose_) std::cout << "[ElectronEnergyCalibrator] ecalEnergy error " << electron.ecalEnergyError() << " recalibrated ecalEnergy error " << newEnergyError_ << std::endl;

 }


void ElectronEnergyCalibrator::computeEpCombination ( const reco::GsfElectron & electron ) 
 {

  //float scEnergy = electron.ecalEnergy() ;
  float scEnergy = newEnergy_ ;
  int elClass = electron.classification() ;

  float trackMomentum  = electron.trackMomentumAtVtx().R() ;
  errorTrackMomentum_ = 999. ;
  
  // retreive momentum error 
  //MultiGaussianState1D qpState(MultiGaussianStateTransform::multiState1D(vtxTsos,0));
  //GaussianSumUtilities1D qpUtils(qpState);
  errorTrackMomentum_ = electron.trackMomentumError();

  float finalMomentum = electron.p4().t(); // initial
  float finalMomentumError = 999.;
  
  // first check for large errors
 
  if (errorTrackMomentum_/trackMomentum > 0.5 && newEnergyError_/scEnergy <= 0.5) {
    finalMomentum = scEnergy;    finalMomentumError = newEnergyError_;
   }
  else if (errorTrackMomentum_/trackMomentum <= 0.5 && newEnergyError_/scEnergy > 0.5){
    finalMomentum = trackMomentum;  finalMomentumError = errorTrackMomentum_;
   }
  else if (errorTrackMomentum_/trackMomentum > 0.5 && newEnergyError_/scEnergy > 0.5){
    if (errorTrackMomentum_/trackMomentum < newEnergyError_/scEnergy) {
      finalMomentum = trackMomentum; finalMomentumError = errorTrackMomentum_;
     }
    else{
      finalMomentum = scEnergy; finalMomentumError = newEnergyError_;
     }
  }
  
  // then apply the combination algorithm
  else {

     // calculate E/p and corresponding error
    float eOverP = scEnergy / trackMomentum;
    float errorEOverP = sqrt(
			     (newEnergyError_/trackMomentum)*(newEnergyError_/trackMomentum) +
			     (scEnergy*errorTrackMomentum_/trackMomentum/trackMomentum)*
			     (scEnergy*errorTrackMomentum_/trackMomentum/trackMomentum));
    //old comb  
//     if ( eOverP  > 1 + 2.5*errorEOverP )
//       {
// 	finalMomentum = scEnergy; finalMomentumError = newEnergyError_;
// 	if ((elClass==reco::GsfElectron::GOLDEN) && electron.isEB() && (eOverP<1.15))
// 	  {
// 	    if (scEnergy<15) {finalMomentum = trackMomentum ; finalMomentumError = errorTrackMomentum_;}
// 	  }
//       }
//     else if ( eOverP < 1 - 2.5*errorEOverP )
//       {
// 	finalMomentum = scEnergy; finalMomentumError = newEnergyError_;
// 	if (elClass==reco::GsfElectron::SHOWERING)
// 	  {
// 	    if (electron.isEB())
// 	      {
// 		if(scEnergy<18) {finalMomentum = trackMomentum; finalMomentumError = errorTrackMomentum_;}
// 	      }
// 	    else if (electron.isEE())
// 	      {
// 		if(scEnergy<13) {finalMomentum = trackMomentum; finalMomentumError = errorTrackMomentum_;}
// 	      }
// 	    else
// 	      { edm::LogWarning("ElectronMomentumCorrector::correct")<<"nor barrel neither endcap electron ?!" ; }
// 	  }
// 	else if (electron.isGap())
// 	  {
// 	    if(scEnergy<60) {finalMomentum = trackMomentum; finalMomentumError = errorTrackMomentum_;}
// 	  }
//       }
//     else 
//       {
// 	// combination
// 	finalMomentum = (scEnergy/newEnergyError_/newEnergyError_ + trackMomentum/errorTrackMomentum_/errorTrackMomentum_) /
// 	  (1/newEnergyError_/newEnergyError_ + 1/errorTrackMomentum_/errorTrackMomentum_);
// 	float finalMomentumVariance = 1 / (1/newEnergyError_/newEnergyError_ + 1/errorTrackMomentum_/errorTrackMomentum_);
// 	finalMomentumError = sqrt(finalMomentumVariance);
//       }
//   }
    
//new comb

    bool eleIsNotInCombination = false ;
     if ( (eOverP  > 1 + 2.5*errorEOverP) || (eOverP  < 1 - 2.5*errorEOverP) || (eOverP < 0.8) || (eOverP > 1.3) )
      { eleIsNotInCombination = true ; }
     if (eleIsNotInCombination)
      {
       if (eOverP > 1)
        { finalMomentum = scEnergy ; finalMomentumError = newEnergyError_ ; }
       else
        {
         if (elClass == reco::GsfElectron::GOLDEN)
          { finalMomentum = scEnergy; finalMomentumError = newEnergyError_; }
         if (elClass == reco::GsfElectron::BIGBREM)
          {
           if (scEnergy<36)
            { finalMomentum = trackMomentum ; finalMomentumError = errorTrackMomentum_ ; }
           else
            { finalMomentum = scEnergy ; finalMomentumError = newEnergyError_ ; }
          }
         if (elClass == reco::GsfElectron::BADTRACK)
          { finalMomentum = scEnergy; finalMomentumError = newEnergyError_ ; }
         if (elClass == reco::GsfElectron::SHOWERING)
          {
           if (scEnergy<30)
            { finalMomentum = trackMomentum ; finalMomentumError = errorTrackMomentum_; }
           else
            { finalMomentum = scEnergy; finalMomentumError = newEnergyError_;}
          }
         if (elClass == reco::GsfElectron::GAP)
          {
           if (scEnergy<60)
            { finalMomentum = trackMomentum ; finalMomentumError = errorTrackMomentum_ ; }
           else
            { finalMomentum = scEnergy; finalMomentumError = newEnergyError_ ; }
          }
        }
      }
 
     else
      {
       // combination
       finalMomentum = (scEnergy/newEnergyError_/newEnergyError_ + trackMomentum/errorTrackMomentum_/errorTrackMomentum_) /
         (1/newEnergyError_/newEnergyError_ + 1/errorTrackMomentum_/errorTrackMomentum_);
       float finalMomentumVariance = 1 / (1/newEnergyError_/newEnergyError_ + 1/errorTrackMomentum_/errorTrackMomentum_);
       finalMomentumError = sqrt(finalMomentumVariance);
      }
  } 


  
// }
  
  math::XYZTLorentzVector oldMomentum = electron.p4() ;
  newMomentum_ = math::XYZTLorentzVector
   ( oldMomentum.x()*finalMomentum/oldMomentum.t(),
     oldMomentum.y()*finalMomentum/oldMomentum.t(),
     oldMomentum.z()*finalMomentum/oldMomentum.t(),
     finalMomentum ) ;
  finalMomentumError_ =  finalMomentumError;  
  //if (verbose_) std::cout << "[ElectronEnergCorrector] old comb momentum " << oldMomentum.t() << " new comb momentum " << newMomentum_.t() << std::endl;

 }

 
//#endif

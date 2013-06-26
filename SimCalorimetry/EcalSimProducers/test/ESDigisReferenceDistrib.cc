/*
 * \file ESDigisReferenceDistrib.cc
 * \ creating reference distribuitons for ES digitization
 *
*/

#include <SimCalorimetry/EcalSimProducers/test/ESDigisReferenceDistrib.h>

ESDigisReferenceDistrib::ESDigisReferenceDistrib(const edm::ParameterSet& ps):
  ESdigiCollection_(ps.getParameter<edm::InputTag>("ESdigiCollection"))
{
  // root and txt outputs
  outputRootFile_ = ps.getUntrackedParameter<std::string>("outputRootFile", "");
  outputTxtFile_  = ps.getUntrackedParameter<std::string>("outputTxtFile", "");
  
  // histos
  char histo[200];
  for (int ii = 0; ii<3 ; ii++) {
    sprintf (histo, "esRefHistos%02d", ii) ;
    meESDigiADC_[ii] = new TH1F(histo, histo, 80, 960.5, 1040.5) ;
  }
  
  meESDigi3D_ = new TH3F("meESDigi3D_", "meESDigi3D_", 80, 960.5, 1040.5, 80, 960.5, 1040.5, 80, 960.5, 1040.5) ;
}

ESDigisReferenceDistrib::~ESDigisReferenceDistrib(){ 

  // preparing the txt file with the histo infos
   ofstream *outFile_ = new ofstream(outputTxtFile_.c_str(),std::ios::out);
  *outFile_ << "# number of bin"                         << std::endl;
  *outFile_ << "# axis inf (common to the three axes"    << std::endl;
  *outFile_ << "# axis sup (common to the three axes"    << std::endl;
  *outFile_ << "# bin x bin content"                     << std::endl;
  *outFile_ << "# "                                      << std::endl;
  
  if(!meESDigi3D_) throw cms::Exception("ESDigisReferenceDistrib: problems with the reference histo");
  else {
    float histoBin_ = meESDigi3D_->GetNbinsX();
    float histoInf_ = meESDigi3D_->GetBinLowEdge(1); 
    float histoSup_ = meESDigi3D_->GetBinLowEdge((int)histoBin_)+meESDigi3D_->GetBinWidth((int)histoBin_);
    
    *outFile_ << histoBin_  << std::endl;  
    *outFile_ << histoInf_  << std::endl;
    *outFile_ << histoSup_  << std::endl;
    
    for (int thisBinZ = 1; thisBinZ <= meESDigi3D_->GetNbinsZ(); thisBinZ++){                  // sample2
      for (int thisBinY = 1; thisBinY <= meESDigi3D_->GetNbinsY(); thisBinY++){                // sample1
	for (int thisBinX = 1; thisBinX <= meESDigi3D_->GetNbinsX(); thisBinX++){	       // sample0    
	  *outFile_ << meESDigi3D_->GetBinContent(thisBinX, thisBinY, thisBinZ) << std::endl;
	}
      }
    }
  }
  
  // saving and deleting
  TFile file(outputRootFile_.c_str(),"RECREATE");
  for (int ii=0; ii<3 ; ii++) { meESDigiADC_[ii] -> Write(); }  
  meESDigi3D_ -> Write();
  file.Close();

  for (int ii=0; ii<3; ii++) { if (meESDigiADC_[ii]){ delete meESDigiADC_[ii]; }}
  if (meESDigi3D_) delete meESDigi3D_;
}

void ESDigisReferenceDistrib::beginJob(){ }

void ESDigisReferenceDistrib::endJob(){ }

void ESDigisReferenceDistrib::analyze(const edm::Event& e, const edm::EventSetup& c){

   edm::Handle<ESDigiCollection> EcalDigiES;  
  e.getByLabel( ESdigiCollection_ , EcalDigiES );
  
  // loop over Digis
  const ESDigiCollection * preshowerDigi = EcalDigiES.product () ;
  
  std::vector<double> esADCCounts ;
  esADCCounts.reserve(ESDataFrame::MAXSAMPLES);
  
  for (unsigned int digis=0; digis<EcalDigiES->size(); ++digis) {

    ESDataFrame esdf=(*preshowerDigi)[digis];
    int nrSamples=esdf.size();

    if ( meESDigi3D_ ) meESDigi3D_ -> Fill(esdf[0].adc(),esdf[1].adc(),esdf[2].adc());
    for (int sample = 0 ; sample < nrSamples; ++sample) {
      ESSample mySample = esdf[sample];
      if (meESDigiADC_[sample]) { meESDigiADC_[sample] ->Fill(mySample.adc()); }
    }
  }
}


//define this as a plug-in

DEFINE_FWK_MODULE(ESDigisReferenceDistrib);

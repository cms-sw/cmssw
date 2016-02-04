#include "Validation/RecoTrack/interface/MTVHistoProducerAlgo.h"

void MTVHistoProducerAlgo::doProfileX(TH2 * th2, MonitorElement* me){
  if (th2->GetNbinsX()==me->getNbinsX()){
    TProfile * p1 = (TProfile*) th2->ProfileX();
    p1->Copy(*me->getTProfile());
    delete p1;
  } else {
    throw cms::Exception("MultiTrackValidator") << "Different number of bins!";
  }
}


void MTVHistoProducerAlgo::fillPlotFromVector(MonitorElement* h, std::vector<int>& vec) {
  for (unsigned int j=0; j<vec.size(); j++){
    h->setBinContent(j+1, vec[j]);
  }
}

void MTVHistoProducerAlgo::fillPlotFromVectors(MonitorElement* h, 
					       std::vector<int>& numerator, 
					       std::vector<int>& denominator,
					       std::string type){
  double value,err;
  for (unsigned int j=0; j<numerator.size(); j++){
    if (denominator[j]!=0){
      if (type=="effic")
	value = ((double) numerator[j])/((double) denominator[j]);
      else if (type=="fakerate")
	value = 1-((double) numerator[j])/((double) denominator[j]);
      else return;
      err = sqrt( value*(1-value)/(double) denominator[j] );
      h->setBinContent(j+1, value);
      h->setBinError(j+1,err);
    }
    else {
      h->setBinContent(j+1, 0);
    }
  }
}




void MTVHistoProducerAlgo::BinLogX(TH1*h){  
  TAxis *axis = h->GetXaxis();
  int bins = axis->GetNbins();
  
  float from = axis->GetXmin();
  float to = axis->GetXmax();
  float width = (to - from) / bins;
  float *new_bins = new float[bins + 1];
  
  for (int i = 0; i <= bins; i++) {
    new_bins[i] = TMath::Power(10, from + i * width);
    
  }
  axis->Set(bins, new_bins);
  delete[] new_bins;
}


//void MTVHistoProducerAlgo::
//void MTVHistoProducerAlgo::


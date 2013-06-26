#include "HcalZeroSuppressionAlgo.h"

HcalZeroSuppressionAlgo::HcalZeroSuppressionAlgo(bool mp) : m_markAndPass(mp) {
   m_dbService=0;
}


void HcalZeroSuppressionAlgo::suppress(const HBHEDigiCollection& input, HBHEDigiCollection& output) {
  HBHEDigiCollection::const_iterator i;

  for (i=input.begin(); i!=input.end(); ++i) 
    if (shouldKeep((*i))) {
      if (!m_markAndPass) output.push_back(*i);
      else {
	HBHEDataFrame df(*i);
	df.setZSInfo(true,false);
	output.push_back(df);
      }
    } else if (m_markAndPass) {
      HBHEDataFrame df(*i);
      df.setZSInfo(true,true);
      output.push_back(df);
    }
}

void HcalZeroSuppressionAlgo::suppress(const HFDigiCollection& input, HFDigiCollection& output) {
  HFDigiCollection::const_iterator i;


  for (i=input.begin(); i!=input.end(); ++i) 
    if (shouldKeep((*i))) {
      if (!m_markAndPass) output.push_back(*i);
      else {
	HFDataFrame df(*i);
	df.setZSInfo(true,false);
	output.push_back(df);
      }
    } else if (m_markAndPass) {
      HFDataFrame df(*i);
      df.setZSInfo(true,true);
      output.push_back(df);
    }
}

void HcalZeroSuppressionAlgo::suppress(const HODigiCollection& input, HODigiCollection& output) {
 HODigiCollection::const_iterator i;


  for (i=input.begin(); i!=input.end(); ++i) 
    if (shouldKeep((*i))) {
      if (!m_markAndPass) output.push_back(*i);
      else {
	HODataFrame df(*i);
	df.setZSInfo(true,false);
	output.push_back(df);
      }
    } else if (m_markAndPass) {
      HODataFrame df(*i);
      df.setZSInfo(true,true);
      output.push_back(df);
    }
}





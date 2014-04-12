#include <iostream>
#include <algorithm> 

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimDataFormats/GeneratorProducts/interface/LHEXMLStringProduct.h"

using namespace edm;
using namespace std;


LHEXMLStringProduct::LHEXMLStringProduct()
{
}

LHEXMLStringProduct::LHEXMLStringProduct(const string& onelheoutput) :
  content_()
{
  content_.push_back(onelheoutput);
}

LHEXMLStringProduct::LHEXMLStringProduct(const vector<string>& manylheoutput) :
  content_()
{
  content_.insert(content_.end(), manylheoutput.begin(), manylheoutput.end());
}

LHEXMLStringProduct::~LHEXMLStringProduct()
{
}

bool LHEXMLStringProduct::mergeProduct(LHEXMLStringProduct const &other)
{
  content_.insert(content_.end(), other.getStrings().begin(), other.getStrings().end());  
  return true;
}


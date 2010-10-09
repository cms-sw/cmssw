#include "HistoData.h"

// for Dump()
#include <iostream>
using namespace std;

HistoData::HistoData(std::string Name, int Number) { number = Number; name = Name; }
HistoData::~HistoData() {}

void HistoData::Dump() {

  cout << "name    = " << name << endl
       << "type    = " << type << endl
       << "x_value = " << x_value << endl
       << "y_value = " << y_value << endl
       << "number  = " << number << endl;

} 

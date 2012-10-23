#include <iostream>
#include <sstream>
#include <math.h>
#include <stdlib.h>
#include "DTUtils.h"

using namespace std;


// There is some debugging to be eliminated

double deltaPhi(double phi1, double phi2) {
  phi1 = (phi1 <0)? (phi1 + 2*pi_greca): phi1;
  phi2 = (phi2 <0)? (phi2 + 2*pi_greca): phi2;
  double dephi = fabs(phi1 - phi2);
  double dist_phi = dephi;
  if(dephi > 2*pi_greca) cout << dephi << " double " << phi1 << " " << phi2 << endl;
  dephi = (dephi > pi_greca)? (2*pi_greca - dephi) :dephi;
  // to check with PLZ
  double dist_phi_max = fabs(phi1 + 2*pi_greca - phi2);
  double dist_phi_min = fabs(phi1 - 2*pi_greca - phi2);
  if(dist_phi_max < dist_phi) dist_phi = dist_phi_max;
  if(dist_phi_min < dist_phi) dist_phi = dist_phi_min;
  if(dist_phi != dephi)
    cout << "double dist_phi = " << dist_phi << " versus " << dephi << endl;
  // end to check with PLZ
  return dephi;
}


float deltaPhi(float phi1, float phi2) {
  phi1 = (phi1 <0)? (phi1 + 2*pi_greca): phi1;
  phi2 = (phi2 <0)? (phi2 + 2*pi_greca): phi2;
  float dephi = fabs(phi1 - phi2);
  float dist_phi = dephi;
  if(dephi > 2*pi_greca) cout << dephi << " float " << phi1 << " " << phi2 << endl;
  dephi = (dephi > pi_greca)? (2*pi_greca - dephi) :dephi;
  // to check with PLZ
  float dist_phi_max = fabs(phi1 + 2*pi_greca - phi2);
  float dist_phi_min = fabs(phi1 - 2*pi_greca - phi2);
  if(dist_phi_max < dist_phi) dist_phi = dist_phi_max;
  if(dist_phi_min < dist_phi) dist_phi = dist_phi_min;
  if(dist_phi != dephi)
    cout << "float dist_phi = " << dist_phi << " versus " << dephi << endl;
  // end to check with PLZ
  return dephi;
}


int deltaPhi(int phi1, int phi2) {
  if(phi1 <0) cout << phi1 << endl;
  if(phi2 <0) cout << phi2 << endl;
  int Ipi_greca = static_cast<int>(pi_greca*4096.);
  phi1 = (phi1 <0)? (phi1 + 2*Ipi_greca): phi1;
  phi2 = (phi2 <0)? (phi2 + 2*Ipi_greca): phi2;
  phi1 = (phi1 > 2*Ipi_greca)? (phi1 - 2*Ipi_greca): phi1;
  phi2 = (phi2 > 2*Ipi_greca)? (phi2 - 2*Ipi_greca): phi2;
  // Indeed there are cases when phi is greater then 2*Ipi_greca: whnce are they from?
  int dephi = abs(phi1 - phi2);
  int dist_phi = dephi;
  if(dephi > 2*Ipi_greca) cout << (float(dephi)/4096) << " short " 
			       << (float(phi1)/4096) << " " << (float(phi2)/4096) 
			       << endl;
  dephi = (dephi > Ipi_greca)? (2*Ipi_greca - dephi) :dephi;
  // to check with PLZ
  int dist_phi_max = abs(phi1 + 2*Ipi_greca - phi2);
  int dist_phi_min = abs(phi1 - 2*Ipi_greca - phi2);
  if(dist_phi_max < dist_phi) dist_phi = dist_phi_max;
  if(dist_phi_min < dist_phi) dist_phi = dist_phi_min;
  if(dist_phi != dephi) {
    cout << "short dist_phi = " << dist_phi << " versus " << dephi << endl;
  }
  // end to check with PLZ
  return dephi;
}


char* biny (int Value, bool truncated_on_the_left , size_t wordSize) {
  const size_t WordSize = wordSize + 1;
  //unsigned int Mask = 0X80000000;

  unsigned int Mask = 1;
  for(size_t i=0; i<(wordSize-1); i++)
    Mask <<= 1;

  int Bit;
  char* Text = new char[WordSize];
  Bit = 0;
  while (Mask != 0) {
    Text [Bit] = ((Mask & Value) == 0) ? '0' : '1';
    Mask >>= 1;
    ++Bit;
  }
  Text [Bit] = 0; // termination character

  if(!truncated_on_the_left)
    return Text;

  for(size_t i=0; i<WordSize; i++)
    cout << Text[i] << flush;
  cout << endl;

  size_t zeros_on_left = 0;
  while(Text[zeros_on_left] == '0') {
    //cout << zeros_on_left << ")\t" << Text[zeros_on_left] << endl;
    zeros_on_left += 1;
  }
  const size_t size = WordSize - zeros_on_left;
  char* FinalText = new char[size];
  for(size_t i=0; i<size; i++)
    FinalText[i] = Text[zeros_on_left + i];
  FinalText[size] = 0;
  return FinalText;
}


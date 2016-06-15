#!/usr/bin/env python

#CHANGE THIS LINE TO CHANGE PU SCENARIO
from SimGeneral.MixingModule import mix_2015_25ns_FallMC_matchData_PoissonOOTPU_cfi as pu

import ROOT
n=len(pu.mix.input.nbPileupEvents.probValue )
x=pu.mix.input.nbPileupEvents.probValue

f= ROOT.TFile("mcpu.root","recreate")
pu = ROOT.TH1F("pileup","pileip",n,0,n)

for i in xrange(0,n) :
   print i
   pu.SetBinContent(i+1,x[i]) 


pu.Write()
f.Write()

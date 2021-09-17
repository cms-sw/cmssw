#! /usr/bin/env python
#
#
# Francisco Yumiceva
# yumiceva@fnal.gov
#
# Fermilab, 2010
#
#____________________________________________________________


from __future__ import print_function
import configparser as ConfigParser
import ROOT
import sys
import math
from array import array

from DataFormats.FWLite import Events, Handle


def main():

    preFnal = 'dcache:/pnfs/cms/WAX/11'
    preCern = 'rfio:/castor/cern.ch/cms'
    pre = preCern
    
    files = [pre+'/store/express/Run2010A/ExpressPhysics/FEVT/v4/000/138/737/FEC10B07-5281-DF11-B3D2-0030487A3DE0.root']
    
    events = Events (files)

    handleBSpot  = Handle ("reco::BeamSpot")

    label    = ("offlineBeamSpot")

    #f = ROOT.TFile("analyzerPython.root", "RECREATE")
    #f.cd()

    #muonPt  = ROOT.TH1F("muonPt", "pt",    100,  0.,300.)
    
    # loop over events
    i = 0
    
    for event in events:
        i = i + 1
        if i%10 == 0:
            print(i)

        event.getByLabel (label, handleBSpot)
        # get the product
        spot = handleBSpot.product()

        print(" x = "+str(spot.x0()))
                        
        if i==10: break
                                                                                    
    #f.Close()

if __name__ == '__main__':
    main()

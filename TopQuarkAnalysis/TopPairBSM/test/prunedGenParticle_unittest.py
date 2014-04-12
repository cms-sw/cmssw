#! /usr/bin/env python

import ROOT
import sys
from DataFormats.FWLite import Events, Handle

files = ["ttbsm_42x_mc.root"]
events = Events (files)
handle1  = Handle ("std::vector<reco::GenParticle>")

# for now, label is just a tuple of strings that is initialized just
# like and edm::InputTag
label1 = ("prunedGenParticles")

f = ROOT.TFile("prunedGenParticle_unittest_fwlite.root", "RECREATE")
f.cd()


# loop over events
i = 0
for event in events:
    i = i + 1
    print  '--------- Processing Event ' + str(i)
    # use getByLabel, just like in cmsRun
    event.getByLabel (label1, handle1)
    # get the product
    gens = handle1.product()


    for igen in range(0,len(gens)) :
        gen = gens[igen]
        if gen.numberOfDaughters() > 1 :
            print '{0:6.0f} : pdgid = {1:6.0f}, status = {2:6.0f}, da0 id = {3:6.0f}, da1 = {4:6.0f}, '.format(
                igen, gen.pdgId(), gen.status(), gen.daughter(0).pdgId(), gen.daughter(1).pdgId()
                )
        elif gen.numberOfDaughters() > 0 :
            print '{0:6.0f} : pdgid = {1:6.0f}, status = {2:6.0f}, da0 id = {3:6.0f}, da1 = {4:6.0f}, '.format(
                igen, gen.pdgId(), gen.status(), gen.daughter(0).pdgId(), -9999
                )
        else :
            print '{0:6.0f} : pdgid = {1:6.0f}, status = {2:6.0f}, da0 id = {3:6.0f}, da1 = {4:6.0f}, '.format(
                igen, gen.pdgId(), gen.status(), -9999, -9999
                )

f.cd()

f.Close()

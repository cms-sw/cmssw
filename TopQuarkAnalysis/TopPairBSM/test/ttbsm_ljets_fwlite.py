#! /usr/bin/env python

from ROOT import gROOT,std,fabs,ROOT,TFile,TTree,TStopwatch,TMatrix,TLorentzVector,TMath,TVector
import sys, math, subprocess
from array import array
from optparse import OptionParser
from ttbsm_ljets_def import *
from DataFormats.FWLite import Events, Handle

# Create a command line option parser
parser = OptionParser()

parser.add_option("--runMuons", action='store_true',
                  default=False,
                  dest="runMuons",
                  help="Electrons(1), Muons(0)")

parser.add_option('--lepPtMin', metavar='F', type='float', action='store',
                  default=15.,
                  dest='lepPtMin',
                  help='lepton pt threshold')

# Parse and get arguments
(options, args) = parser.parse_args()
runMu = options.runMuons
lepPtMin = options.lepPtMin

files = ["patTuple_tlbsm_train_ttbar.root"]
printGen = True
events = Events (files)

jetsH       = Handle ("std::vector<pat::Jet>")
muonsH      = Handle ("std::vector<pat::Muon>")
electronsH  = Handle ("std::vector<pat::Electron>")
metsH       = Handle("std::vector<pat::MET>")
# for now, label is just a tuple of strings that is initialized just like and edm::InputTag
jetsLabel      = ("goodPatJets")
muonsLabel     = ("selectedPatMuons")
electronsLabel = ("selectedPatElectrons")
metsLabel      = ("patMETPF")

f = TFile("outplots.root", "RECREATE")
f.cd()

t = TTree("tree", "tree")

max_nLeps = 2
max_nJets = 30

nleps = array('i',[0])
t.Branch('nleps',nleps,'nLeps/I')

njets = array('i',[0])
t.Branch('njets',njets,'nJets/I')

if runMu:
    isMuTight =  array('d',max_nLeps*[0.])
    t.Branch('isMuTight',isMuTight,'isMuTight[nLeps]/D')
else:
    eMVA = array('d', max_nLeps*[0.])
    t.Branch('eMVA',eMVA,'eMVA[nLeps]/D')
    
lepIso = array('d',max_nLeps*[0.])
t.Branch('lepIso',lepIso,'lepIso[nLeps]/D')

lepE = array('d',max_nLeps*[0.])
t.Branch('lepE',lepE,'lepE[nLeps]/D')

lepPx = array('d',max_nLeps*[0.])
t.Branch('lepPx',lepPx,'lepPx[nLeps]/D')

lepPy = array('d',max_nLeps*[0.])
t.Branch('lepPy',lepPy,'lepPy[nLeps]/D')

lepPz = array('d',max_nLeps*[0.])
t.Branch('lepPz',lepPz,'lepPz[nLeps]/D')

jetE = array('d',max_nJets*[0.])
t.Branch('jetE',jetE,'jetE[nJets]/D')

jetPx = array('d',max_nJets*[0.])
t.Branch('jetPx',jetPx,'jetPx[nJets]/D')

jetPy = array('d',max_nJets*[0.])
t.Branch('jetPy',jetPy,'jetPy[nJets]/D')

jetPz = array('d',max_nJets*[0.])
t.Branch('jetPz',jetPz,'jetPz[nJets]/D')

met = array('d',[0.])
t.Branch('met',met,'met/D')

# Keep some timing information
nEventsAnalyzed = 0
timer = TStopwatch()
timer.Start()

# loop over events
i = 0
for event in events:
    i = i + 1
    if i % 100 == 0 :
        print  '--------- Processing Event ' + str(i)

    nEventsAnalyzed = nEventsAnalyzed + 1

    ##---------reset for next event---------
    for l in range(max_nLeps):
        lepE[l] = 0.0; lepPx[l] = 0.0; lepPy[l] = 0.0; lepPz[l] = 0.0; lepIso[l] = -1.0;
        if runMu: isMuTight[l] = 0.0
        else: eMVA[l] = 0.0
        
    for j in range(max_nJets):
        jetE[j] = 0.0; jetPx[j] = 0.0; jetPy[j] = 0.0; jetPz[j] = 0.0;

    ##------------Get the handles -----------
    event.getByLabel(muonsLabel, muonsH)
    event.getByLabel(electronsLabel, electronsH)
    event.getByLabel (jetsLabel,  jetsH)
    event.getByLabel (metsLabel,  metsH)

    if muonsH.isValid():
        muons = muonsH.product()
    if electronsH.isValid():
        electrons = electronsH.product()

    # If neither muons nor electrons are found, skip    
    if  muonsH.isValid() and electronsH.isValid():
        if len(muons) == 0 and len(electrons) == 0:
            continue
        
    # assigning one varaible to handle both lepton flavours
    if runMu:
        leptons = muons
    else:
        leptons = electrons
        
    # If no leptons are found, skip
    if len(leptons) == 0 :
        continue

    # If no jets are found, skip    
    if jetsH.isValid():
        jets = jetsH.product()
        if len(jets) == 0: continue

    # Now get the MET
    if metsH.isValid():
        met = metsH.product()[0]


    
    # Count number of leptons passing preselection
    nl = 0    
    for ilep in leptons:        
        if ilep.pt() <= lepPtMin: continue
        #passLoose = 0
        if runMu:
            passLoose = abs(ilep.eta()) < 2.4 and isLooseMu(ilep)           
        else:
            passLoose = fabs(ilep.eta()) < 2.5 and ilep.passConversionVeto()          
        if not passLoose : continue
        #print [a for a in dir(ilep.pfIsolationVariables())]
        # fill lepton P4
        lepE[nl] = ilep.E()
        #check isolation
        chIso = ilep.pfIsolationVariables().sumChargedHadronPt
        nhIso = ilep.pfIsolationVariables().sumNeutralHadronEt
        phIso = ilep.pfIsolationVariables().sumPhotonEt
        puIso = ilep.pfIsolationVariables().sumPUPt
        pfIso = (chIso + max(0.0, nhIso + phIso - 0.5*puIso))/ilep.pt()
        print 'before resetting the event: ', lepIso[nl]
        lepIso[nl] = pfIso
        nl = nl + 1
        print 'iso', pfIso, 'nLep = ', nl, 'iso stored = ', lepIso[nl]

   
    nleps[0] = nl
    #check if there are leptons passing the minimal ID requirments
    #if nLep == 0: continue
    
    t.Fill()
    print 'new event'

timer.Stop()

# Print out our timing information
rtime = timer.RealTime(); # Real time (or "wall time")
ctime = timer.CpuTime(); # CPU time
print("Analyzed events: {0:6d}".format(nEventsAnalyzed))
print("RealTime={0:6.2f} seconds, CpuTime={1:6.2f} seconds".format(rtime,ctime))
print("{0:4.2f} events / RealTime second .".format( nEventsAnalyzed/rtime))
print("{0:4.2f} events / CpuTime second .".format( nEventsAnalyzed/ctime))
subprocess.call( ["ps aux | grep skhalil | cat > memory.txt", ""], shell=True )
##     nElectrons = 0
##     if electronsH.isValid():
##         for iel in electrons:
##             if iel.pt() > lepPtMin:
##                 nElectrons += 1

 

    #print 'nLep = ', nLeptons
    
    #print '---- ' + jetsLabel


##     ijet = 0
##     for jet in jets0 :
##         print ("Jet {0:4.0f}, pt = {1:10.2f}, eta = {2:6.2f}, phi = {3:6.2f}, m = {4:6.2f}, " +
##                "nda = {5:3.0f}, vtxmass = {6:6.2f}, area = {7:6.2f}, L1 = {8:6.2f}, L2 = {9:6.2f}, L3 = {10:6.2f}, " +
##                "currLevel = {11:s}").format(
##             ijet, jet.pt(), jet.eta(), jet.phi(), jet.mass(), jet.numberOfDaughters(), jet.userFloat('secvtxMass'),
##             jet.jetArea(), jet.jecFactor("L1FastJet"), jet.jecFactor("L2Relative"), jet.jecFactor("L3Absolute"), jet.currentJECLevel()
##             ),
##         if printGen :
##             genPt = 0.
##             if jet.genJetFwdRef().isNonnull() and jet.genJetFwdRef().isAvailable() :
##                 genPt = jet.genJetFwdRef().pt()
##             else :
##                 genPt = -1.0
##             print (", gen pt = {0:6.2f}").format( genPt )
##         else :
##             print ''
##         ijet += 1
    
##     print '---- ' + label1
##     # use getByLabel, just like in cmsRun
##     event.getByLabel (label1, handle1)
##     # get the product
##     jets1 = handle1.product()

##     ijet = 0
##     for jet in jets1 :
##         print 'Jet {0:4.0f}, pt = {1:10.2f}, eta = {2:6.2f}, phi = {3:6.2f}, m = {4:6.2f}, nda = {5:3.0f}'.format(
##             ijet, jet.pt(), jet.eta(), jet.phi(), jet.mass(), jet.numberOfDaughters()
##             ),
##         if jet.numberOfDaughters() > 1 :
##             print ', ptda1 = {0:6.2f}, ptda1 = {1:6.2f}'.format( jet.daughter(0).pt(), jet.daughter(1).pt() )
##         else :
##             print ''
##         ijet += 1


##     print '---- ' + label2
##     # use getByLabel, just like in cmsRun
##     event.getByLabel (label2, handle2)
##     # get the product
##     jets2 = handle2.product()

##     ijet = 0
##     for jet in jets2 :
##         print 'Jet {0:4.0f}, pt = {1:10.2f}, eta = {2:6.2f}, phi = {3:6.2f}, m = {4:6.2f}, nda = {5:3.0f}, topmass = {6:6.2f}, minmass = {7:6.2f}'.format(
##             ijet, jet.pt(), jet.eta(), jet.phi(), jet.mass(), jet.numberOfDaughters(), jet.tagInfo('CATop').properties().topMass, jet.tagInfo('CATop').properties().minMass
##             )
##         ijet += 1

        
##     print '---- ' + label3
##     # use getByLabel, just like in cmsRun
##     event.getByLabel (label3, handle3)
##     # get the product
##     jets3 = handle3.product()

##     ijet = 0
##     for jet in jets3 :
##         print 'Jet {0:4.0f}, pt = {1:10.2f}, eta = {2:6.2f}, phi = {3:6.2f}, m = {4:6.2f}, nda = {5:3.0f}'.format(
##             ijet, jet.pt(), jet.eta(), jet.phi(), jet.mass(), jet.numberOfDaughters()
##             ),
##         if jet.numberOfDaughters() > 2 :
##             print ', ptda1 = {0:6.2f}, ptda1 = {1:6.2f}, ptda2 = {2:6.2f}'.format( jet.daughter(0).pt(), jet.daughter(1).pt(), jet.daughter(2).pt() )
##         else :
##             print ''
##         ijet += 1


##     print '---- ' + label4
##     # use getByLabel, just like in cmsRun
##     event.getByLabel (label4, handle4)
##     # get the product
##     muons1 = handle4.product()

##     imuon = 0
##     for muon in muons1 :
##         if not muon.isGlobalMuon() :
##             continue
##         print 'Muon {0:4.0f}, pt = {1:10.2f}, eta = {2:6.2f}, phi = {3:6.2f}, m = {4:6.2f}, nda = {5:3.0f}, chi2/dof = {6:6.2f}'.format(
##             imuon, muon.pt(), muon.eta(), muon.phi(), muon.mass(), muon.numberOfDaughters(), muon.normChi2()
##             )
##         imuon += 1

##     print '---- ' + label5
##     # use getByLabel, just like in cmsRun
##     event.getByLabel (label5, handle5)
##     # get the product
##     electrons1 = handle5.product()

##     ielectron = 0
##     for electron in electrons1 :
##         print 'Electron {0:4.0f}, pt = {1:10.2f}, eta = {2:6.2f}, phi = {3:6.2f}, m = {4:6.2f}, nda = {5:3.0f}, eidTight = {6:6.2f}'.format(
##             ielectron, electron.pt(), electron.eta(), electron.phi(), electron.mass(), electron.numberOfDaughters(), electron.electronID('eidTight')
##             )
##         ielectron += 1 


# "cd" to our output file
f.cd()

# Write our tree
t.Write()

# Close it
f.Close()

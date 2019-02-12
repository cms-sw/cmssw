# Author: Juska Pekkanen, juska@cern.ch
#
# Run as '$python validor.py (int)maxevents (str)file_suffix 
#                            (str)data_type ("data" or "mc") (str)datatier ("naod")'
# example: $python validor.py 100 testrun mc naod

import IO # Not used 'yet'
import Jet # Important self-written class in Jet.py

# Not all of the below packages necessarily used *TODO*
import ROOT as r
import params as p
import sys
import numpy as np
import scipy as sc
import array as ar
import bisect as bs


def main():

   
   # Define cmd line inputs & parsing
   limit = 0
   if len(sys.argv) > 1:
      limit = int(sys.argv[1])
   if len(sys.argv) > 2:
      suffix = sys.argv[2]
   else:
      suffix = ""

   if "data" in sys.argv:
      isMC = False
   elif "mc" in sys.argv:
      isMC = True
   else:
      print "Fatal error: data type not specified."
      sys.exit()
      
   if "naod" in sys.argv:
      dtier = "naod"
   elif "djred" in sys.argv:
      dtier = "djred"
   else:
      dtier = None
      print "Fatal error: unknown data tier '%s'." % dtier
      sys.exit()
    

   curdir = r.gDirectory

   # Choose the input data format. For new formats, please implement a new
   # input definition to 'Jet.py' and other future classes
   # For now (Dec 2018) NanoAOD is the standard input format, but 'djred' from
   # dijet search is left to live as an example for adding new formats 

   if (dtier == "naod"):
      tree = r.TChain("Events")
      tree.Add("/home/juska/jet_tuples/nanoaod/pfdev/"
         #"test_mc_102X_NANO_hadd_ken_flatqcd_original.root")
         "test_mc_102X_NANO_hadd_ken_flatqcd_v2_rawECAL_hadcal.root")
      
      fout = r.TFile("res/validor_naod_%s_%i.root" % (suffix, limit), "recreate")
      
   elif (dtier == "djred"):   
   
      tree = r.TChain("rootTupleTree/tree")
      tree.AddFile("myfile.root")
      fout = r.TFile("res/classyjet_djred_%s_%i.root" % (suffix, limit), "recreate")
      
   else:
      print "Fatal error: unknown data tier '%s'." % dtier
      sys.exit()   
      
   
   # Define variable binning with ar.array()
   
   ptbins = ar.array('d',[10,24,32,43,56,74,97,133,174,245,300,362,430,
                  507,592,686,846,1032,1248,1588,2000,2500,3000,4000,6000])
   nptbins = len(ptbins)-1
   
   ptbinsfine = ar.array('d',[10 ,24, 28,
     32, 37, 43, 49, 56, 64, 74, 84, 97, 114, 133, 153, 174, 196, 220,
      245, 272, 300, 330, 362, 395, 430, 468, 507, 548, 592, 638, 686, 737, 790,
      846, 905, 967, 1032, 1101, 1172, 1248, 1327, 1410, 1497, 1588, 1684, 1784,
      1890, 2000])
   nptbinsfine = len(ptbinsfine)-1
   
   etabins = ar.array('d',[0,1.3,1.653,2.172,5.205])
   netabins = len(etabins)-1
   etabinsfine = ar.array('d',[-5.205,-4.903,-4.73,-4.552,-4.377,-4.027,-3.853,-3.677,
                  -3.503,-3.327,-3.152,-3.000,-2.650,-2.322,-2.043,-1.830,
                  -1.653,-1.479,-1.305,-1.131,-0.957,-0.783,-0.522,-0.261,
                  0,0.261,0.522,0.783,0.957,1.131,1.305,1.479,1.653,1.830,
                  2.043,2.322,2.650,3.000,3.152,3.327,3.503,3.677,3.853,4.027,
                  4.377,4.552,4.730,4.903,5.205])
   netabinsfine = len(etabinsfine)-1
      

   # Jet energy fractions and flavors and dictionaries for handy filling
   
   lfracs = ["chf", "nhf", "cemf", "nemf","muf"]
   lprops = ["qgl","nelec","nmuon","nconst"]
   dflavs = {"u":1,"d":2,"s":3,"c":4,"b":5,"t":6,"g":21,"uds":123}

   dfracflav = {}
   dpropflav = {}
   
   for frac in lfracs:
      for flav in dflavs.keys():
         dfracflav["%s_%s" % (frac,flav)] = dflavs[flav]
         
   for prop in lprops:
      for flav in dflavs.keys():
         dpropflav["%s_%s" % (prop,flav)] = dflavs[flav]
         
         
   # Create directories to clarify file reading
   
   dir_ptfracs = fout.mkdir("pt_fracs")
   dir_etafracs = fout.mkdir("eta_fracs")
   dir_ptfracsflav = fout.mkdir("pt_fracs_flav")
   dir_etafracsflav = fout.mkdir("eta_fracs_flav")
   dir_2d = fout.mkdir("2D_fracs")
   dir_simple = fout.mkdir("simple")
   dir_other = fout.mkdir("other")
   dir_respo = fout.mkdir("respo")
   
   
   # Prepare profile dictionaries for pt and eta slices
   dfracpt={}; dfraceta={}; dflavpt={}; dflaveta={}; d2dbin={}; d2dgev={}
   dfracpt_nojetid={}; dfraceta_nojetid={}
   dfracpt_manjetid={}; dfraceta_manjetid={}
   dfracpt_mikkojetid={}; dfraceta_mikkojetid={}
   
   for frac in lfracs:
      proflistpt = []; proflisteta = []
      proflistpt2 = []; proflisteta2 = []
      proflistpt3 = []; proflisteta3 = []
      proflistpt4 = []; proflisteta4 = []
      dir_ptfracs.cd()
      for i in range(netabins):
         name = "%s_eta_slice_%1.1f-%1.1f" % (frac,etabins[i],etabins[i+1])
         name2 = "%s_eta_slice_nojid_%1.1f-%1.1f" % (frac,etabins[i],etabins[i+1])
         name3 = "%s_eta_slice_manjid_%1.1f-%1.1f" % (frac,etabins[i],etabins[i+1])
         name4 = "%s_eta_slice_mikjid_%1.1f-%1.1f" % (frac,etabins[i],etabins[i+1])
         proflistpt.append(r.TProfile(name,name,nptbins,ptbins))
         proflistpt2.append(r.TProfile(name2,name2,nptbins,ptbins))
         proflistpt3.append(r.TProfile(name3,name3,nptbins,ptbins))
         proflistpt4.append(r.TProfile(name4,name4,nptbins,ptbins))
      dfracpt[frac] = proflistpt
      dfracpt_nojetid[frac] = proflistpt2
      dfracpt_manjetid[frac] = proflistpt3
      dfracpt_mikkojetid[frac] = proflistpt4
      dir_etafracs.cd()
      for i in range(nptbins):
         name = "%s_pt_slice_%1.1f-%1.1f" % (frac,ptbins[i],ptbins[i+1])
         name2 = "%s_pt_slice_nojid_%1.1f-%1.1f" % (frac,ptbins[i],ptbins[i+1])
         name3 = "%s_pt_slice_manjid_%1.1f-%1.1f" % (frac,ptbins[i],ptbins[i+1])
         name4 = "%s_pt_slice_mikjid_%1.1f-%1.1f" % (frac,ptbins[i],ptbins[i+1])         
         proflisteta.append(r.TProfile(name,name,netabinsfine,etabinsfine))
         proflisteta2.append(r.TProfile(name2,name2,netabinsfine,etabinsfine))
         proflisteta3.append(r.TProfile(name3,name3,netabinsfine,etabinsfine))
         proflisteta4.append(r.TProfile(name4,name4,netabinsfine,etabinsfine))
      dfraceta[frac] = proflisteta
      dfraceta_nojetid[frac] = proflisteta2
      dfraceta_manjetid[frac] = proflisteta3
      dfraceta_mikkojetid[frac] = proflisteta4 
      

   # 2D profiles for future projections (pun intended)
   dir_2d.cd()
   for frac in lfracs:
      name = "%s_2D_binned" % frac
      d2dbin[frac] = r.TProfile2D(name,name,nptbins,ptbins,netabinsfine,etabinsfine)
      
   # Flavour fractions
   for frac in lfracs:
      for flav in dflavs.keys():
         proflistpt = []; proflisteta = []
         dir_ptfracsflav.cd()
         for i in range(netabins):
            name = "%s_%s_eta_slice_%1.1f-%1.1f" % (frac,flav,etabins[i],etabins[i+1])
            proflistpt.append(r.TProfile(name,name,nptbins,ptbins))
         dflavpt["%s_%s"%(frac,flav)] = proflistpt
         dir_etafracsflav.cd()
         for i in range(nptbins):
               name = "%s_%s_e_slice_%1.1f-%1.1f" % (frac,flav,ptbins[i],ptbins[i+1])
               proflisteta.append(r.TProfile(name,name,netabinsfine,etabinsfine))
         dflaveta["%s_%s"%(frac,flav)] = proflisteta
   
   # Simple fraction histos (better name to be invented)
   dfracsimple = {}
   dpropsimple = {}
   dir_simple.cd()

   # Flavs vs fractions
   for frac in lfracs:
      for flav in dflavs.keys():
         name = "%s_%s_simple" % (frac,flav)
         dfracsimple["%s_%s"%(frac,flav)] = r.TH1D(name,name,18,0,1)

   # Flavs vs properties
   for prop in lprops:
      for flav in dflavs.keys():
         name = "%s_%s_simple" % (prop,flav)
         if prop == "qgl":
            dpropsimple["%s_%s"%(prop,flav)] = r.TH1D(name,name,100,-1,1)
         elif prop == "nconst":
            dpropsimple["%s_%s"%(prop,flav)] = r.TH1D(name,name,50,0,200)
         else:
            dpropsimple["%s_%s"%(prop,flav)] = r.TH1D(name,name,10,0,10)
         
   dir_other.cd()
      
   nbins = 200
   
   # Parton-wise
   hchf_g = r.TH1F("CHF_g", "CHF_g", nbins, 0, 1)
   hchf_u = r.TH1F("CHF_u", "CHF_u", nbins, 0, 1)
   hchf_b = r.TH1F("CHF_b", "CHF_b", nbins, 0, 1)
   hchf_s = r.TH1F("CHF_s", "CHF_s", nbins, 0, 1)
   
   
   # Profiles for response and resolution
   dir_respo.cd()      
   prespbar = r.TProfile("prespbar",
      "<P_{T}^{PF}>/<P_{T}^{Gen}>, barrel",nptbinsfine,ptbinsfine);
   prespend = r.TProfile("prespend",
      "<P_{T}^{PF}>/<P_{T}^{Gen}>, endcaps",nptbinsfine,ptbinsfine);
   presptra = r.TProfile("presptra",
      "<P_{T}^{PF}>/<P_{T}^{Gen}>, trans.",nptbinsfine,ptbinsfine);
   prespall = r.TProfile("prespall",
      "<P_{T}^{PF}>/<P_{T}^{Gen}> everywhere",nptbinsfine,ptbinsfine);
      
      
   # Store a histogram for each pT bin for then measuring resolution, i.e.
   # the standard deviation of this distribution in each bin.
   # Tnp stands for tag-and-probe.
   aresoeta05 = []; aresoeta05tnp = []
   aresoeta13 = []; aresoeta13tnp = []
   aresoeta13_21 = []; aresoeta13_21tnp = []
   aresoeta21_25 = []; aresoeta21_25tnp = []
   aresoeta25_30 = []; aresoeta25_30tnp = []
   
   # Ugly I know. It's Monday.
   for idx in range(len(ptbins)-1):
      ptlow = int(ptbins[idx]); pthi = int(ptbins[idx+1])
      
      name = "reso_dist_%i_%i_eta05" % (ptlow,pthi)
      name2 = "reso_dist_%i_%i_eta05_tnp" % (ptlow,pthi)
      aresoeta05.append(r.TH1D(name,name,200,0,2))
      aresoeta05tnp.append(r.TH1D(name2,name2,200,0,2))
      
      name = "reso_dist_%i_%i_eta13" % (ptlow,pthi)
      name2 = "reso_dist_%i_%i_eta13_tnp" % (ptlow,pthi)
      aresoeta13.append(r.TH1D(name,name,200,0,2))
      aresoeta13tnp.append(r.TH1D(name2,name2,200,0,2))
     
      name = "reso_dist_%i_%i_eta13_21" % (ptlow,pthi)
      name2 = "reso_dist_%i_%i_eta13_21_tnp" % (ptlow,pthi)
      aresoeta13_21.append(r.TH1D(name,name,200,0,2))
      aresoeta13_21tnp.append(r.TH1D(name2,name2,200,0,2))
      
      name = "reso_dist_%i_%i_eta21_25" % (ptlow,pthi)
      name2 = "reso_dist_%i_%i_eta21_25_tnp" % (ptlow,pthi)
      aresoeta21_25.append(r.TH1D(name,name,200,0,2))
      aresoeta21_25tnp.append(r.TH1D(name2,name2,200,0,2))
      
      name = "reso_dist_%i_%i_eta25_30" % (ptlow,pthi)
      name2 = "reso_dist_%i_%i_eta25_30_tnp" % (ptlow,pthi)
      aresoeta25_30.append(r.TH1D(name,name,200,0,2))
      aresoeta25_30tnp.append(r.TH1D(name2,name2,200,0,2))
      
      
   # Event loop starting
   
   entries = tree.GetEntries()   
   print "Events in tree: %i" % entries
   print "Starting analysis with event count %i" % min(entries,limit)
   
   prec_check = []
   for idx, event in enumerate(tree):
   
   
      if (idx % 200000 == 0 and idx > 0):
         print "Events processed: %i" % idx
      
      if limit > 0 and idx > limit: # Do this somehow nicer...
         break
         
      # Skip obvious recostruction errors
      if event.nJet > 0 and event.Jet_pt[0] > 6500:
         print "Skipping event with lead-jet pt " + str(event.Jet_pt[0])
         continue
         
      # Skip events without possible dijet/TnP topology FIXME check logic
      if (event.nJet < 2):
         continue
      
      # Same for genjets for response
      if (event.nGenJet < 2):
         continue
         
         
      j = Jet.Jet(event, 0, isMC, dtier)
      j2 = Jet.Jet(event, 1, isMC, dtier)
      # 3rd jet pt needed below
      if (event.nJet > 2):
         j3pt = event.Jet_pt[2]
      else:
         j3pt = 0
         
      # Check tight jet ID and skip if needed
      if (j.jetidt == False or j2.jetidt == False):
         continue  
         
         
      # Check back-to-back and soft 3rd jet conditions for dijet topology
      if (abs(j.phi - j2.phi) > 2.8 and 0.3*(j.pt+j2.pt)/2.0 > j3pt):
         b2b = True
      else:
         b2b = False
         
         
      weight = event.Generator_weight

      # Jet loop
      for ijet in range(min(event.nJet,event.nGenJet)):
         
         # For tag-and-probe. Not used everywhere.
         if (ijet == 0):
            jtag = j; jpro = j2
         elif (ijet == 1):
            jtag = j2; jpro = j
         elif (ijet > 1):
            jtag = None; jpro = None
         else:
            print "Logic is broken."
            sys.exit()
            
         # jdm for jet-direct-match. To be used with all non-TnP studies.
         jdm = Jet.Jet(event,ijet,isMC,dtier)
               
         # Fill stuff for response and resolution
         # Tag-and-probe implemented, need to double-check logic *TODO*
         dr = None
         ratio = None
         if (ijet < 2 and b2b):
            vreco = r.TLorentzVector()
            vgen = r.TLorentzVector()
            # I guess I have to ensure matching for probe, not tag
            # (DeltaR could be got faster doing it manually)
            vreco.SetPtEtaPhiM(jpro.pt,jpro.eta,jpro.phi,jpro.mass)
            vgen.SetPtEtaPhiM(jpro.genpt,jpro.geneta,jpro.genphi,jpro.genmass)
            dr = vreco.DeltaR(vgen)


         vrecodm = r.TLorentzVector()
         vgendm = r.TLorentzVector()

         vrecodm.SetPtEtaPhiM(jdm.pt,jdm.eta,jdm.phi,jdm.mass)
         vgendm.SetPtEtaPhiM(jdm.genpt,jdm.geneta,jdm.genphi,jdm.genmass)
         drdm = vrecodm.DeltaR(vgendm)

         # Fill response profiles after genjet matching
         # (this is withOUT TnP now)
         if (drdm<0.2): # and ijet < 2 and b2b):
            resp = jdm.rawpt/jdm.genpt
            #prespall.Fill(jdm.genpt,resp,weight)
            prespall.Fill(jdm.genpt,resp,1)
            if jdm.genabseta < 1.3:
               prespbar.Fill(jdm.genpt,resp)
            if jdm.genabseta > 1.3 and jdm.genabseta < 2.4:
               prespend.Fill(jdm.genpt,resp)
               
         # Fill resolution histos after genjet matching
         if (dr<0.2 and ijet < 2 and b2b):
         
            idxtnp = bs.bisect(ptbins,jtag.pt)-1
            ratiotnp = jpro.pt/vgen.Pt()
         
            # Skip pt under/overflow
            if (jtag.pt < ptbins[0] or jtag.pt > ptbins[-1]):
               print "Overflow tag jet with pt " + str(jtag.pt)
               continue
               
            # Tag-and-probe
            if (jtag.abseta < 0.5):
               aresoeta05tnp[idxtnp].Fill(ratiotnp)
               
            if (jtag.abseta < 1.3):
               aresoeta13tnp[idxtnp].Fill(ratiotnp)

            if (jtag.abseta > 1.3 and jtag.abseta < 2.1):
               aresoeta13_21tnp[idxtnp].Fill(ratiotnp)

            if (jtag.abseta > 2.1 and jtag.abseta < 2.5):
               aresoeta21_25tnp[idxtnp].Fill(ratiotnp)
               
            if (jtag.abseta > 2.5 and jtag.abseta < 3.0):
               aresoeta25_30tnp[idxtnp].Fill(ratiotnp)

         # DM resolution histos
         if (drdm<0.2):
         
            idx = bs.bisect(ptbins,vgendm.Pt())-1
            ratio = jdm.pt/vgendm.Pt()
                                             
            # Direct match (arbitrary choice between jtag and jpro)
            # Question: should I use gen-eta here in the cut instead? Nope, reco!
            if (jdm.abseta < 0.5):
               aresoeta05[idx].Fill(ratio)
               
            if (jdm.abseta < 1.3):
               aresoeta13[idx].Fill(ratio)

            if (jdm.abseta > 1.3 and jdm.abseta < 2.1):
               aresoeta13_21[idx].Fill(ratio)

            if (jdm.abseta > 2.1 and jdm.abseta < 2.5):
               aresoeta21_25[idx].Fill(ratio)
               
            if (jdm.abseta > 2.5 and jdm.abseta < 3.0):
               aresoeta25_30[idx].Fill(ratio)
               
         
         #Stick to tag-and-probe from this on
         if (not b2b & ijet > 1):
            continue
            
         # Skip events with tag out of barrel for compo studies
         if abs(jtag.eta) > 1.3:
            continue
         
         
         # Frac dict for filling. Now use probe jet.
         dfracs = {"chf":jpro.chf,"nhf":jpro.nhf,"cemf":jpro.cemf,"nemf":jpro.nemf,\
            "muf":jpro.muf}
         # Prop dict for filling
         dprops = {"qgl":jpro.qgl,"nelec":jpro.nelec,"nmuon":jpro.nmuon,"nconst":jpro.nconst}
         if len(dfracs.keys()) != len(lfracs):
            print "Fatal error: fraction count mismatch. Bye."
            sys.exit()               
         
         # Fill fraction dictionaries
         
         for frac in dfracs.keys():
            binidx = bs.bisect(etabins,jpro.abseta)
            # Bisect starts from 1
            if (binidx > len(dfracpt[frac])):
               print "Warning: eta overflow with tag jet eta ", jtag.abseta, "skipping!"
               continue
            
            dfracpt[frac][binidx-1].Fill(jtag.pt,dfracs[frac])
            
            # Flav info only in MC
            if isMC:
               # Fill flavs-fracs vs pt
               iflav = j.pflav
               for flav in dflavs.keys():
                  name = "%s_%s" % (frac,flav)
                  if iflav == dfracflav[name]:
                     dflavpt[name][binidx-1].Fill(j.pt,dfracs[frac])
                     
                     # Fill simple histos for barrel only
                     if j.abseta < 1.3:
                        dfracsimple[name].Fill(dfracs[frac])
                        if (iflav < 4):
                           dfracsimple["%s_uds"%frac].Fill(dfracs[frac])

         
         for prop in dprops.keys():
            iflav = j.pflav
            for flav in dflavs.keys():
               name = "%s_%s" % (prop,flav)
               if iflav == dpropflav[name]:
                  if j.abseta < 1.3:
                     dpropsimple[name].Fill(dprops[prop])
                     if (iflav < 4):
                        dpropsimple["%s_uds"%prop].Fill(dprops[prop])
         
         
         for frac in dfracs.keys():
            binidx = bs.bisect(ptbins,jpro.pt)
            if (binidx > len(dfraceta[frac])): # Add pT bins if this is frequent!
               print "Warning: pt overflow with tag jet pt ", jtag.pt, "skip!"
               continue

            if jtag.jetidt == True:
               dfraceta[frac][binidx-1].Fill(jtag.eta,dfracs[frac])
               d2dbin[frac].Fill(jtag.pt,jtag.eta,dfracs[frac])
               
   fout.cd()
   fout.Write()
   
   print "Done. Output written to %s" % fout.GetName()
   
#      raw_input("Press enter to exit")
   fout.Close() # Close only here to see plots
      
main()

# Runtime with 82.5k events MC: 2m8.480s

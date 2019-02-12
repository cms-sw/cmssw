import IO
import sys

class Jet(IO.Event):

   def __init__(self, evt, idx, isMC, dtier):
   
      # Jet NanoAOD
      if (dtier == "naod"):
         self._idx = idx
         self._pt = evt.Jet_pt[idx]
         self._rawpt = evt.Jet_pt[idx]*(1-evt.Jet_rawFactor[idx])
         self._isMC = isMC
         self._eta = evt.Jet_eta[idx]
         self._phi = evt.Jet_phi[idx]
         self._mass = evt.Jet_mass[idx]
         self._abseta = abs(self._eta)
         self._area = evt.Jet_area[idx]
         self._cemf = evt.Jet_chEmEF[idx]
         self._nemf = evt.Jet_neEmEF[idx]
         self._chf = evt.Jet_chHEF[idx]
         self._nhf = evt.Jet_neHEF[idx]
         
         self._jetid = evt.Jet_jetId[idx]
         self._rawfac = evt.Jet_rawFactor[idx]
         self._nconst = evt.Jet_nConstituents[idx]
         self._nelec = evt.Jet_nElectrons[idx]
         self._nmuon = evt.Jet_nMuons[idx]
         self._bgtags = [evt.Jet_btagCMVA[idx],evt.Jet_btagCSVV2[idx],
                         evt.Jet_btagDeepB[idx],
                         evt.Jet_btagDeepC[idx]]#evt.Jet_btagDeepFlavB[idx]]
         self._qgl = evt.Jet_qgl[idx]

         # Vars only in MC
         if self._isMC:
            self._pflav = evt.Jet_partonFlavour[idx]
            self._hflav = evt.Jet_hadronFlavour[idx]
            
            self._genpt = evt.GenJet_pt[idx]
            self._geneta = evt.GenJet_eta[idx]
            self._genphi = evt.GenJet_phi[idx]
            self._genmass = evt.GenJet_mass[idx]
            self._genabseta = abs(self._geneta)
            

            
            
         # True muon fraction only in Sal's data for now
         # (Except for now it is in official data too, thanks to meeee!!)   
         #if not self._isMC:
         self._muf = evt.Jet_muEF[idx]

         # Calculate tight jetID boolean
         foo = [4,5,6,7] # Hack for getting ID from binary
         # (someone please tell me how to do it less ugly!)
         
         
         # This is now TightLeptonVetoID
         if (evt.Jet_jetId[idx] not in foo):
            self._jetidt = False
         else:
            self._jetidt = True
            
      elif (dtier == "djred"):
      
         self._idx = idx
         self._pt = evt.pTWJ_j1 if idx == 0 else evt.pTWJ_j2
         self._isMC = isMC
         self._eta = evt.Jet_eta[idx]
         self._abseta = abs(self._eta)
         self._area = evt.Jet_area[idx]
         self._cemf = evt.Jet_chEmEF[idx]
         self._nemf = evt.Jet_neEmEF[idx]
         self._chf = evt.Jet_chHEF[idx]
         self._nhf = evt.Jet_neHEF[idx]
         self._muf = evt.Jet_muEF[idx]

      
      else:
         print "Fatal error: unknown data tier '%s'. Cya kook" % dtier
         sys.exit()
   
   # Getters
   @property
   def idx(self):
      return self._idx
   @property
   def isMC(self):
      return self._isMC
   @property
   def pt(self):
      return self._pt
   @property
   def rawpt(self):
      return self._rawpt
   @property
   def phi(self):
      return self._phi
   @property
   def area(self):
      return self._area
   @property
   def cemf(self):
      return self._cemf
   @property
   def nemf(self):
      return self._nemf
   @property
   def chf(self):
      return self._chf
   @property
   def nhf(self):
      return self._nhf
   @property
   def muf(self):
      return self._muf
   @property
   def jetidt(self):
      return self._jetidt
   @property
   def rawfac(self):
      return self._rawfac
   @property
   def jetidt(self):
      return self._jetidt
   @property
   def eta(self):
      return self._eta
   @property
   def abseta(self):
      return self._abseta
   @property
   def nconst(self):      
      return self._nconst
   @property
   def nelec(self):
      return self._nelec
   @property
   def nmuon(self):
      return self._nmuon
   @property
   def btags(self):
      return self._btags
   @property
   def qgl(self):
      return self._qgl
   @property
   def mass(self):
      return self._mass

   # MC-only vars need special care
   @property
   def genpt(self):
      if self._isMC:
         return self._genpt
      else:
         return None
   @property
   def geneta(self):
      if self._isMC:
         return self._geneta
      else:
         return None
   @property
   def genphi(self):
      if self._isMC:
         return self._genphi
      else:
         return None
   @property
   def genmass(self):
      if self._isMC:
         return self._genmass
      else:
         return None
   @property
   def genabseta(self):
      if self._isMC:
         return self._genabseta
      else:
         return None
   @property
   def pflav(self):
      if self._isMC:
         return self._pflav
      else:
         return None
      
      
   # Setters
   @pt.setter
   def pt(self, val):
      self._pt = val

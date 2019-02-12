# This file contains classes and methods for basic input and ouput operations
# for the TeraJet jet particology package.

import ROOT as r

# Not used yet, let's see if we need one later
class Event:
   
   def __init__(self, evt):
      #self.__filename = filename
      self._njet = evt.nJet
      self._eventno = evt.event
      self._run = evt.run
      self._lumi = evt.luminosityBlock

   # Getters
   @property
   def njet(self):
      return self._njet
   @property
   def eventno(self):
      return self._eventno
   @property
   def run(self):
      return self._run
   @property
   def lumi(self):
      return self._lumi
      
   #def load_file(self, filename):
   #   self.__filename = filename
   #   f = r.TFile(filename,"read")
   
   #def load_vars(self, evt):
   #   self.njet = evt.nJet
   #   self.
      

      
def load_file(dtype, filename):
   
   f = r.TFile(filename, "read")
   if dtype == "SMPJ":
      treepath = "ak5/ProcessedTree"
   elif dtype == "NANO":
      treepath = "Events"
   else:
      print "Fatal error: no known tree path for data type '%s'" % dtype
      exit()
   tree = r.TTree()
   tree = f.Get(treepath)

   # Thanks ROOT/C++, do this to return an object with it's original type
   tree.SetDirectory(0)
   
   #tree.AddDirectory(kFalse)
   #r.gInterpreter.ProcessLine('TTree::AddDirectory(kFalse)')
   entries = tree.GetEntriesFast()
   
   return tree
   
   

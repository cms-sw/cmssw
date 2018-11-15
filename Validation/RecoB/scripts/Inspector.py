from __future__ import print_function
#____________________________________________________________
#
#  cuy
#
# A very simple way to make plots with ROOT via an XML file
#
# Francisco Yumiceva
# yumiceva@fnal.gov
#
# Fermilab, 2008
#
#____________________________________________________________

import sys
import ROOT
from ROOT import TFile


class Inspector:

    def SetFilename(self, value):
        self.Filename = value
    def Verbose(self, value):
        self.Verbose = value

    def createXML(self, value):
        self.XML = value

    def SetTag(self,value):
        self.tag = value
        self.TagOption = True

    def Loop(self):

        afile = TFile(self.Filename)
        afilename = self.Filename
        stripfilename = afilename

        try:
            if self.TagOption:
                stripfilename = self.tag
        except:
            stripfilename = afilename.split('/')[len(afilename.split('/')) -1]
            stripfilename = stripfilename[0:(len(stripfilename)-5)]

        alist = self.dir.GetListOfKeys()

        for i in alist:
            aobj = i.ReadObj()
            if aobj.IsA().InheritsFrom("TDirectory"):
                if self.Verbose:
                    print(' found directory: '+i.GetName())

                if self.XML:
                    print('   <!-- '+i.GetName()+' -->')

                bdir = self.dir
                afile.GetObject(i.GetName(),bdir)
                blist = bdir.GetListOfKeys()
                for j in blist:
                    bobj = j.ReadObj()
                    if bobj.IsA().InheritsFrom(ROOT.TH1.Class()):
                        if self.Verbose:
                            print('  --> found TH1: name = '+j.GetName() + ' title = '+j.GetTitle())
                        if self.XML:
                            print('   <TH1 name=\"'+stripfilename+'_'+j.GetName()+'\" source=\"'+'/'+i.GetName()+'/'+j.GetName()+'\"/>')

    def GetListObjects(self):

        afile = TFile(self.Filename)

        if afile.IsZombie():
            print(" error trying to open file: " + self.Filename)
            sys.exit()

        if self.XML:

            print('''
<cuy>
''')	
            print('  <validation type=\"'+afile.GetName()+'\" file=\"'+self.Filename+'\" release=\"x.y.z\">')

        self.dir = ROOT.gDirectory
        self.Loop()

        if self.XML:

            print('''
  </validation>

</cuy>
''')










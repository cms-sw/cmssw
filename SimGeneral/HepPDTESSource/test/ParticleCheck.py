#Need to write proper documentation!
'''
This script is designed to read the G4msg.log output of a cmsDriver.py command run in a test release with a special configuration of SimG4Core/Application/RunManager.cc and a special version of SimulationG4.py fragment.
To dump the G4 particle table one needs to:
1-Edit SimG4Core/Application/src/RunManager.cc:
  a-Add in the includes at the top:
    #include "G4ParticleTable.hh"
  b-Add at the bottom of the RunManager::initG4 method:
    edm::LogInfo("SimG4CoreApplication") << "Output of G4ParticleTable DumpTable:";
    G4ParticleTable::GetParticleTable()->DumpTable("ALL");
2-Edit the Validation/Performance/python/TimeMemoryG4Info.py customise fragment (or you could create your own):
  a-Configure the output (in this case to the file G4msg.log) to include SimG4CoreApplication:
    process.MessageLogger.files = dict(G4msg =  cms.untracked.PSet(
      noTimeStamps = cms.untracked.bool(True)
      #First eliminate unneeded output
      ,threshold = cms.untracked.string('INFO')
      ,INFO = cms.untracked.PSet(limit = cms.untracked.int32(0))
      ,FwkReport = cms.untracked.PSet(limit = cms.untracked.int32(0))
      ,FwkSummary = cms.untracked.PSet(limit = cms.untracked.int32(0))
      ,Root_NoDictionary = cms.untracked.PSet(limit = cms.untracked.int32(0))
      ,FwkJob = cms.untracked.PSet(limit = cms.untracked.int32(0))
      ,TimeReport = cms.untracked.PSet(limit = cms.untracked.int32(0))
      ,TimeModule = cms.untracked.PSet(limit = cms.untracked.int32(0))
      ,TimeEvent = cms.untracked.PSet(limit = cms.untracked.int32(0))
      ,MemoryCheck = cms.untracked.PSet(limit = cms.untracked.int32(0))
      #TimeModule, TimeEvent, TimeReport are written to LogAsbolute instead of LogInfo with a category
      #so they cannot be eliminated from any destination (!) unless one uses the summaryOnly option
      #in the Timing Service... at the price of silencing the output needed for the TimingReport profiling
      #
      #Then add the wanted ones:
      ,PhysicsList = cms.untracked.PSet(limit = cms.untracked.int32(-1))
      ,G4cout = cms.untracked.PSet(limit = cms.untracked.int32(-1))
      ,G4cerr = cms.untracked.PSet(limit = cms.untracked.int32(-1))
      ,SimG4CoreApplication = cms.untracked.PSet(limit = cms.untracked.int32(-1))
      )
    )
3-Run any cmsDriver.py commands that entail simulation, e.g.(in CMSSW_3_1_0_pre4):
  cmsDriver.py MinBias.cfi -n 1 --step GEN,SIM --customise=Validation/Performance/TimeMemoryG4Info.py --eventcontent FEVTDEBUG --conditions FrontierConditions_GlobalTag,IDEAL_30X::All > & ! MinBias.log &
The resulting file G4msg.log contains the dump of the G4 Particle Table. We run on it, extract the information we are interested in and we store it in 2 dictionaries:
G4ParticleTable and G4ParticleTablePDG, the first one using Particle Name as keys, the second one using PDG ID as keys.
The script also reads the HepPDT particle table contained in /src/SimGeneral/HepPDTESSource/data/pythiaparticle.tbl, it extracts all the information and stores it in 2 equivalent dictionaries,HepPdtTable and HepPdtTablePDG.
Then some comparisons are made.
Starting from the HepPdtTablePDG (since in CMSSW the handshake is via the PDG ID code) and look if that PDG ID exists in the G4. If it does we then check:
1-For particles with ctau>10mm we check if G4 has decay tables for them (as it should)
2-For all particles (that matched PDG ID) we check that they have:
 a-the same mass
 b-the same charge
 c-the same ctau
3-For the ones that don''t match we dump some output files with the information
We also dump the list of particles in HepPdtTablePDG that do not find a PDG ID match in G4ParticleTablePDG.

'''
G4msg=open('G4msg.log','r')

#Two functions to dump output into files
def WriteOut(myfile,mylist):
    #Need to write the help/documentation for the function
    mylist=map(lambda a:str(a)+' ',mylist)
    mylist+='\n'
    myfile.writelines(mylist)

def WriteOutHtml(myfile,mylist):
    #Need to write the help/documentation for the function
    myoutputline=['<tr>']
    mylist=map(lambda a:'<td align="center">'+str(a)+' '+'</td>',mylist)
    mylist+=['<tr>\n']
    myoutputline+=mylist
    myfile.writelines(mylist)

G4cout=False
NewParticle=False
G4ParticleTable={}
G4ParticleTablePDG={}
ParticleName=''
ParticleInfoDict={}
pdgcode=0
#antipdgcode=0
PDGcode=''
#Speed of light in mm/nsec (to translate G4 lifetime from nsec to HepPdt ctau in mm) 
c=299.792458
Tokens=[]
for record in G4msg:
    if G4cout: #Handle only G4 messages
        if record[0:4]!='%MSG': #Getting rid of all the Message logger stuff (each line of the table is a G4cout message!)
            if '--- G4ParticleDefinition ---' in record: #Delimiter for a single particle information
                NewParticle=True
            if NewParticle: #Write the old one in and get the next info
                #First a dictionary based on ParticleName as keys
                G4ParticleTable.update({ParticleName:ParticleInfoDict})
                #Then the most useful one, by PDG ID.
                G4ParticleTablePDG.update({pdgcode:ParticleInfoDict})
                #To make it more easily comparable to HepPDT, duplicate entry for antiparticle (G4 has only 1 entry for particle and antiparticle)
                #G4ParticleTablePDG.update({antipdgcode:ParticleInfoDict})
                ParticleInfoDict={}
            NewParticle=False
            tokens=map(lambda a:a.strip(' \n'),record.split(':'))
            if 'Particle Name' in tokens[0]:
                ParticleName=tokens[1]
                ParticleInfoDict.update({'Particle Name':ParticleName})
            if 'PDG particle code' in tokens[0]:
                pdgcode=int(tokens[1].split()[0])
                ParticleInfoDict.update({'PDG ID':pdgcode})
            if 'Mass [GeV/c2]' in tokens[0]:
                mass=float(tokens[1].split()[0])
                ParticleInfoDict.update({'Mass [GeV/c2]':mass})
            if 'Charge [e]' in tokens[0]:
                charge=float(tokens[1])
                ParticleInfoDict.update({'Charge [e]':charge})
            if 'Lifetime [nsec]' in tokens[0]:
                lifetime=float(tokens[1])
                ParticleInfoDict.update({'Lifetime [nsec]':lifetime})
                if lifetime!=-1:
                    ctau=c*lifetime
                    ParticleInfoDict.update({'ctau [mm]':ctau})
                else:
                    ParticleInfoDict.update({'ctau [mm]':-999999999})
            if 'G4DecayTable' in tokens[0]:
                decaytable=True
                ParticleInfoDict.update({'G4 Decay Table':decaytable})
            if 'Decay Table is not defined !!' in tokens[0]:
                decaytable=False
                ParticleInfoDict.update({'G4 Decay Table':decaytable})
        elif record[0:5]=='%MSG ': #Notice the space
            #print 'G4 message ended'
            G4cout=False
    if record[0:13]=='%MSG-i G4cout':
        #print 'G4 message started'
        G4cout=True 
import os
import math
HepPdtTable={}
HepPdtTablePDG={}
for record in open(os.environ.data['CMSSW_RELEASE_BASE']+'/src/SimGeneral/HepPDTESSource/data/pythiaparticle.tbl','r'):
    if '//' in record:
        pass
    else:
        tokens = record.split()
        #print tokens
        if len(tokens)==6: #There is an empty line at the end of the file
            #Note the different conventions (Particle names are different, charge is reported in units of 1/3e originally in HepPDT, lifetime expressed in ctau in cm)
            HepPdtTable.update({tokens[1]:{'Particle Name':tokens[1],'PDG ID':int(tokens[0]),'Charge [e]': float(tokens[2])/3, 'Mass [GeV/c2]':float(tokens[3]), 'ctau [mm]':float(tokens[5])}})
            HepPdtTablePDG.update({int(tokens[0]):{'Particle Name':tokens[1],'PDG ID':int(tokens[0]),'Charge [e]': float(tokens[2])/3, 'Mass [GeV/c2]':float(tokens[3]), 'ctau [mm]':float(tokens[5])}})

#A few consistency checks on the dictionaries
#G4ParticleTables
#Quick and dirty removal of empty first element of G4ParticleTable ('':{}) should really fix the code, but for now it's OK...
print 'Popping the first empty element of G4ParticleTable: %s '%G4ParticleTable.pop('')
print 'G4ParticleTable contains ',len(G4ParticleTable),'elements'
print 'G4ParticleTablePDG contains ',len(G4ParticleTablePDG),'elements'
if len(G4ParticleTable)>len(G4ParticleTablePDG):
    print "The following values were in the G4ParticleTable dictionary but not in the G4ParticleTablePDG one (multiple entries with different names but the same PDG code):"
    for value in G4ParticleTable.values():
        if value not in G4ParticleTablePDG.values():
            print value
elif len(G4ParticleTablePDG)>len(G4ParticleTable):
    print "The following values were in the G4ParticleTablePDG dictionary but not in the G4ParticleTable one (multiple entries with different PDG codes but the same Particle Name):"
    for value in G4ParticleTablePDG.values():
        if value not in G4ParticleTable.values():
            print value
print 'HepPdtTable contains ',len(HepPdtTable),'elements'
print 'HepPdtTablePDG contains ',len(HepPdtTablePDG),'elements'
if len(HepPdtTable)>len(HepPdtTablePDG):
    print "The following values were in the HepPdtTable dictionary but not in the HepPdtTablePDG one (multiple entries with different names but the same PDG code):"
    for value in HepPdtTable.values():
        if value not in HepPdtTablePDG.values():
            print value
elif len(HepPdtTablePDG)>len(HepPdtTable):
    print "The following values were in the HepPdtTablePDG dictionary but not in the HepPdtTable one (multiple entries with different PDG codes but the same Particle Name):"
    for value in HepPdtTablePDG.values():
        if value not in HepPdtTable.values():
            print value

#Comparison via dictionaries!
        
#Let's try to do the test the way things work in our code (SimG4Core/Application/src/Generators.cc):
#Start from HepPdtTable with PDG code. This is the only thing we pass to G4, with the momentum, so
#Need to check if Mass, Charge are the same. And for a handful of particles with ctau>10mm that GEANT has the decay table.

MatchingPDG=[]
MatchingPDGMass=[]
MassDiffPercentList=[]
ctauDiffPercentList=[]
NotMatchingPDG=[]
#Prepare some html output
MatchingPDGDecayFileHtml=open('MatchingPDGDecay.html','w')
MatchingPDGDecayFileHtml.writelines(['<html>\n','<body>\n','<table align="center", border=2>\n'])
WriteOutHtml(MatchingPDGDecayFileHtml,['PDG ID', 'HepPdt Particle Name','G4 Particle Name',' HepPdt ctau [mm]','G4 ctau [mm]',' HepPdt Mass [GeV/c2]','G4 Mass [GeV/c2]',' G4 Decay Table'])
MatchingPDGChargeFileHtml=open('MatchingPDGCharge.html','w')
MatchingPDGChargeFileHtml.writelines(['<html>\n','<body>\n','<table align="center", border=2>\n'])
WriteOutHtml(MatchingPDGChargeFileHtml,['PDG ID', 'HepPdt Particle Name','G4 Particle Name',' HepPdt Charge [e]','G4 Charge [e]'])
NoLifeTimePDGFileHtml=open('NoLifeTimePDGG4.html','w')
NoLifeTimePDGFileHtml.writelines(['<html>\n','<body>\n','<table align="center", border=2>\n'])
WriteOutHtml(NoLifeTimePDGFileHtml,['PDG ID', 'HepPdt Particle Name','G4 Particle Name','HepPdt ctau [mm]','G4 ctau [mm]',' HepPdt Mass [GeV/c2]','G4 Mass [GeV/c2]'])

#MatchingPDGMassFileHtml=open('MatchingPDGMass.html','w')
#MatchingPDGMassFileHtml.writelines(['<html>\n','<body>\n','<table align="center", border=2>\n'])
#WriteOutHtml(MatchingPDGMassFileHtml,['PDG ID', 'HepPdt Particle Name','G4 Particle Name',' HepPdt Mass [GeV/c2]','G4 Mass [GeV/c2]','Mass Diff [GeV/c2]','Mass Diff %'])
#MatchingPDGctauFileHtml=open('MatchingPDGctau.html','w')
#MatchingPDGctauFileHtml.writelines(['<html>\n','<body>\n','<table align="center", border=2>\n'])
#WriteOutHtml(MatchingPDGctauFileHtml,['PDG ID', 'HepPdt Particle Name','G4 Particle Name',' HepPdt ctau [mm]','G4 ctau [mm]','ctau Diff [mm]','ctau Diff %'])

for pdgcode in HepPdtTablePDG.keys():
    if pdgcode in G4ParticleTablePDG.keys():
        #Save the PDG code matching particles Particle Name in MatchingPDG list
        MatchingPDG+=[HepPdtTablePDG[pdgcode]['Particle Name']]
        ######################################################################################
        #Start checks for particles that can be passed to G4, since we pass them by PDG codes!
        ######################################################################################
        #Check that G4 has decay tables for particles with ctau>10 mm:
        if G4ParticleTablePDG[pdgcode]['ctau [mm]']>10.0 or HepPdtTablePDG[pdgcode]['ctau [mm]']>10.0:
            if not G4ParticleTablePDG[pdgcode]['G4 Decay Table']:
                print "****Uh Oh No G4 Decay Table for ", G4ParticleTablePDG[pdgcode]['Particle Name']
            else:
                WriteOutHtml(MatchingPDGDecayFileHtml,[pdgcode,HepPdtTablePDG[pdgcode]['Particle Name'],G4ParticleTablePDG[pdgcode]['Particle Name'],HepPdtTablePDG[pdgcode]['ctau [mm]'],G4ParticleTablePDG[pdgcode]['ctau [mm]'],HepPdtTablePDG[pdgcode]['Mass [GeV/c2]'],G4ParticleTablePDG[pdgcode]['Mass [GeV/c2]'],G4ParticleTablePDG[pdgcode]['G4 Decay Table']])
        #Since we are here also compare ctaus
        if G4ParticleTablePDG[pdgcode]['ctau [mm]']==-999999999:
            WriteOutHtml(NoLifeTimePDGFileHtml,[pdgcode,HepPdtTablePDG[pdgcode]['Particle Name'],G4ParticleTablePDG[pdgcode]['Particle Name'],HepPdtTablePDG[pdgcode]['ctau [mm]'],G4ParticleTablePDG[pdgcode]['ctau [mm]'],HepPdtTablePDG[pdgcode]['Mass [GeV/c2]'],G4ParticleTablePDG[pdgcode]['Mass [GeV/c2]']])
            G4ParticleTablePDG[pdgcode]['ctau [mm]']=0.0
        if HepPdtTablePDG[pdgcode]['ctau [mm]']!=G4ParticleTablePDG[pdgcode]['ctau [mm]']:
            ctauDiff=HepPdtTablePDG[pdgcode]['ctau [mm]']-G4ParticleTablePDG[pdgcode]['ctau [mm]']
            if HepPdtTablePDG[pdgcode]['ctau [mm]']!=0:
                ctauDiffPercent=math.fabs(ctauDiff/HepPdtTablePDG[pdgcode]['ctau [mm]']*100)
                ctauDiffPercentList+=[(abs(ctauDiffPercent),pdgcode,ctauDiff,ctauDiffPercent)]
            elif G4ParticleTablePDG[pdgcode]['ctau [mm]']!=0:
                ctauDiffPercent=math.fabs(ctauDiff/G4ParticleTablePDG[pdgcode]['ctau [mm]']*100)
                ctauDiffPercentList+=[(abs(ctauDiffPercent),pdgcode,ctauDiff,ctauDiffPercent)]
            else:
                ctauDiffPercent=0.0
                ctauDiffPercentList+=[(abs(ctauDiffPercent),pdgcode,ctauDiff,ctauDiffPercent)]
                
            #WriteOutHtml(MatchingPDGctauFileHtml,[pdgcode, HepPdtTablePDG[pdgcode]['Particle Name'],G4ParticleTablePDG[pdgcode]['Particle Name'], HepPdtTablePDG[pdgcode]['ctau [mm]'],G4ParticleTablePDG[pdgcode]['ctau [mm]'],ctauDiff,ctauDiffPercent])
        #Check Mass
        if HepPdtTablePDG[pdgcode]['Mass [GeV/c2]']!=G4ParticleTablePDG[pdgcode]['Mass [GeV/c2]']:
            MassDiff=HepPdtTablePDG[pdgcode]['Mass [GeV/c2]']-G4ParticleTablePDG[pdgcode]['Mass [GeV/c2]']
            MassDiffPercent=math.fabs(MassDiff/HepPdtTablePDG[pdgcode]['Mass [GeV/c2]']*100)
            MassDiffPercentList+=[(abs(MassDiffPercent),pdgcode,MassDiff,MassDiffPercent)]
            print pdgcode,HepPdtTablePDG[pdgcode]['Particle Name'],G4ParticleTablePDG[pdgcode]['Particle Name'],' Mass:',HepPdtTablePDG[pdgcode]['Mass [GeV/c2]'],' Mass:',G4ParticleTablePDG[pdgcode]['Mass [GeV/c2]'],MassDiff,MassDiffPercent
            #WriteOutHtml(MatchingPDGMassFileHtml,[pdgcode,HepPdtTablePDG[pdgcode]['Particle Name'],G4ParticleTablePDG[pdgcode]['Particle Name'],HepPdtTablePDG[pdgcode]['Mass [GeV/c2]'],G4ParticleTablePDG[pdgcode]['Mass [GeV/c2]'],MassDiff,MassDiffPercent])
        else:
            #Save the PDG code and Mass matching particles Particle Name in MatchingPDGMass list
            MatchingPDGMass+=[HepPdtTablePDG[pdgcode]['Particle Name']]
        #Check Charge
        if HepPdtTablePDG[pdgcode]['Charge [e]']!=G4ParticleTablePDG[pdgcode]['Charge [e]']:
            print pdgcode,HepPdtTablePDG[pdgcode]['Particle Name'],G4ParticleTablePDG[pdgcode]['Particle Name'],' Charge [e]:',HepPdtTablePDG[pdgcode]['Charge [e]'],' Charge [e]:',G4ParticleTablePDG[pdgcode]['Charge [e]']
            WriteOutHtml(MatchingPDGChargeFileHtml,[pdgcode,HepPdtTablePDG[pdgcode]['Particle Name'],G4ParticleTablePDG[pdgcode]['Particle Name'],HepPdtTablePDG[pdgcode]['Charge [e]'],G4ParticleTablePDG[pdgcode]['Charge [e]']])
    else:
        #########################################################################################
        #Start checks for particles that cannot be passed to G4, since we pass them by PDG codes!
        #########################################################################################
        NotMatchingPDG+=[(HepPdtTablePDG[pdgcode]['ctau [mm]'],pdgcode,HepPdtTablePDG[pdgcode]['Particle Name'])]
        
        
#Sort the list of percentual mass difference (absolute value), PDG from the highest discrepancy to the lowest
#In case we want to dump a table with the particle names etc
MatchingPDGMassSortedFileHtml=open('MatchingPDGMassSorted.html','w')
MatchingPDGMassSortedFileHtml.writelines(['<html>\n','<body>\n','<table align="center", border=2>\n'])
WriteOutHtml(MatchingPDGMassSortedFileHtml,['PDG ID', 'HepPdt Particle Name','G4 Particle Name',' HepPdt Mass [GeV/c2]','G4 Mass [GeV/c2]','Mass Diff [GeV/c2]','Mass Diff %','G4 ctau [mm]'])
MassDiffPercentList.sort(reverse=True)
for element in MassDiffPercentList:
    pdgcode=element[1]
    MassDiff=element[2]
    MassDiffPercent=element[3]
    WriteOutHtml(MatchingPDGMassSortedFileHtml,[pdgcode,HepPdtTablePDG[pdgcode]['Particle Name'],G4ParticleTablePDG[pdgcode]['Particle Name'],HepPdtTablePDG[pdgcode]['Mass [GeV/c2]'],G4ParticleTablePDG[pdgcode]['Mass [GeV/c2]'],MassDiff,MassDiffPercent,G4ParticleTablePDG[pdgcode]['ctau [mm]']])
#    print element[0],element[1]

MatchingPDGctauSortedFileHtml=open('MatchingPDGctauSorted.html','w')
MatchingPDGctauSortedFileHtml.writelines(['<html>\n','<body>\n','<table align="center", border=2>\n'])
WriteOutHtml(MatchingPDGctauSortedFileHtml,['PDG ID', 'HepPdt Particle Name','G4 Particle Name',' HepPdt ctau [mm]','G4 ctau [mm]','ctau Diff [mm]','ctau Diff %'])
ctauDiffPercentList.sort(reverse=True)
for element in ctauDiffPercentList:
    pdgcode=element[1]
    ctauDiff=element[2]
    ctauDiffPercent=element[3]
    WriteOutHtml(MatchingPDGctauSortedFileHtml,[pdgcode, HepPdtTablePDG[pdgcode]['Particle Name'],G4ParticleTablePDG[pdgcode]['Particle Name'], HepPdtTablePDG[pdgcode]['ctau [mm]'],G4ParticleTablePDG[pdgcode]['ctau [mm]'],ctauDiff,ctauDiffPercent])


#Print/Write out the NotMatchingPDG elements PDG code Particle Name (HepPdtTable convention)
NotMatchingPDGFileHtml=open('NotMatchingPDG.html','w')
NotMatchingPDGFileHtml.writelines(['<html>\n','<body>\n','<table align="center", border=2>\n'])
WriteOutHtml(NotMatchingPDGFileHtml,['PDG ID', 'HepPdt Particle Name','HepPdt Mass [GeV/c2]',' HepPdt Charge [e]','HepPdt ctau [mm]'])
NotMatchingPDG.sort(reverse=True)
for element in NotMatchingPDG:
    pdgcode=element[1]
    WriteOutHtml(NotMatchingPDGFileHtml,[pdgcode,HepPdtTablePDG[pdgcode]['Particle Name'],HepPdtTablePDG[pdgcode]['Mass [GeV/c2]'],HepPdtTablePDG[pdgcode]['Charge [e]'],HepPdtTablePDG[pdgcode]['ctau [mm]']])

#Closing files etc:
MatchingPDGDecayFileHtml.writelines(['</table>\n','<body>\n','</html>\n'])
MatchingPDGChargeFileHtml.writelines(['</table>\n','<body>\n','</html>\n'])
MatchingPDGMassSortedFileHtml.writelines(['</table>\n','<body>\n','</html>\n'])
MatchingPDGctauSortedFileHtml.writelines(['</table>\n','<body>\n','</html>\n'])
NotMatchingPDGFileHtml.writelines(['</table>\n','<body>\n','</html>\n'])
NoLifeTimePDGFileHtml.writelines(['</table>\n','<body>\n','</html>\n'])

#MatchingPDGMassFileHtml.writelines(['</table>\n','<body>\n','</html>\n'])
#MatchingPDGctauFileHtml.writelines(['</table>\n','<body>\n','</html>\n'])

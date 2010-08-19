#!/usr/bin/env python


ecaltower = 1  
hcaltower = 2  
electronthr = 90
clusterthr = 3

isoratioEA = 6
isoratioEB = 50
isoratioTA = 10
isoratioTB = 50
isozone  =3
jetc = 2
jetet=30

carddim = 64
eta0=-32
etam=+32
phi0=-3
phim=76


print "<root>"
print "<CARD>"
print ""
print ""

print "<SETTINGS ECALTower = \"%d\" HCALTower = \"%d\" ElectronThr = \"%d\" ClusterThr = \"%d\" IsolationElectronA = \"%d\" IsolationElectronB = \"%d\" IsolationTauA = \"%d\" IsolationTauB = \"%d\" IsolationZone = \"%d\" JetCenter = \"%d\" JetEt = \"%d\" />"%(ecaltower,hcaltower,electronthr,clusterthr,isoratioEA,isoratioEB,isoratioTA,isoratioTB,isozone,jetc,jetet)
print ""
print ""
print ""
print ""


print "<GEOMETRY eta0 = \"%d\" phi0 = \"%d\" dim = \"%d\" etam = \"%d\" phim = \"%d\"     />"%(4,4,carddim,60,75) 

wire=0
iphi=phi0;
while iphi<=phim:
   phi=iphi
   if phi==-3:
      phi=69
   if phi==-2:
      phi=70
   if phi==-1:
      phi=71
   if phi==0:
      phi=72
   if phi==73:
      phi=1
   if phi==74:
      phi=2
   if phi==75:
      phi=3
   if phi==76:
      phi=4
   eta=eta0   
   while eta<=etam:
      if eta!=0:
         if eta>=-28 and eta<=28: 
            print "<WIRE no = \"%d\" eta = \"%d\" phi = \"%d\"/>"%(wire,eta,phi)

         wire=wire+1   
      eta=eta+1

   iphi=iphi+1      

print "</CARD>"
print "</root>"


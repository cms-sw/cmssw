#!/usr/bin/env python
#A script to test threading concepts/implementations
import threading,time,sys
#The following used from past bad experience with multithreading in Python
def _cleanup():
    pass

#Define the "thread" class, of which each thread is a instance
class TestThread(threading.Thread):
    #Constructor with 2 arguments
    def __init__(self,Name,Cpu,**kwargs): #Name and Cpu are equivalent to any number of options necessary to instantiate the Test()
        self.Name=Name
        self.Cpu=Cpu
        self.kwargs=kwargs
        threading.Thread.__init__(self)
    #Actual function executed at the invocation of thread's start() method    
    def run(self):
        #Instantiate the test of class Test()
        self.Test=Test(self.Name,self.Cpu,**(self.kwargs))
        #This is the function used to really activate the test
        #Launch it!
        self.Test.runTest()
        return

#Define the class Test, of which each test (executed in a thread or not) is an instance
class Test:
    #Constructor with 2 optional arguments
    def __init__(self,Name="N/A",Cpu="N/A",**kwargs):
        self.Name=Name
        self.Cpu=Cpu
        self.kwargs=kwargs
        #Initializing some list to keep timestamps for a silly test
        self.Times=[]
        print "Initializing Test() instance, value of Name is %s and valud of Cpu is %s"%(self.Name,self.Cpu)

    #Silly functions to get back the Name and Cpu arguments originally passed to the Test object
    def getName(self):
        return self.Name

    def getCpu(self):
        return self.Cpu
    #The actual test function
    def runTest(self):
        print "I am thread Test and I was invoked with arguments Name %s, Cpu %s and optional keyword arguments %s"%(self.Name,self.Cpu,self.kwargs)
        self.time=0
        while self.time<10:
            self.Times.append(time.ctime())
            time.sleep(1)
            self.time+=1
        print self.Times
        if self.kwargs:
            print "Testing keyword arguments handling with function invocation"
            test(**(self.kwargs))
        return
    
#Test function for arguments fun
ahi="AHI!"
def test(cpu='N/A',perfsuitedir=ahi,IgProfEvents='N/A',IgProfCandles='N/A',cmsdriverOptions='N/A',stepOptions='N/A',string="IgProf",profilers='N/A',bypasshlt='N/A',userInputFile='N/A'):
    print cpu
    print perfsuitedir
    print userInputFile
    #print "Value of Available is: %s"%Available

#Playing with classes for variable scope tests:
class Pippo:
    def __init__(self):
        self.a=0
        self.b=1
    def test1(self,d):
        print d
    def test2(self):
        self.e=self.Pluto(self)
        self.e.testscope()
    class Pluto:
        def __init__(self,mother):
            self.Me="Pluto"
            self.mother=mother
        def testscope(self):
            #print a
            #print self.a
            self.mother.test1(self.Me)
def main():
    #Testing threading concepts ;)
    #First set that all 4 cores are available:
    Available=['0','1','2','3']
    #Then populate the list of tests to do:
    #This list should be a list of arguments with which to run simpleGenReport (except the cpu).
    TestToDo=['Pippo','Pluto','Paperino','Minnie','Qui','Quo','Qua','Zio Paperone','Banda Bassotti','Archimede','Topolino'] 
    
    #Now let's set up an infinite loop that will go through the TestToDo list, submit a thread per cpu available from the Available list
    #using pop.
    activeThreads={}
    while 1:
        #If there are cores available and tests to run:
        print "Main while loop:"
        print Available
        print TestToDo
        #Logic based on checking for TestToDo first:
        if TestToDo:
            print "Still folllowing %s tests to do:"%len(TestToDo)
            print TestToDo
            #Test available cores:
            if Available:
                print "Hey there is at least one core available!"
                print Available
                cpu=Available.pop()
                print "Let's use core %s"%cpu
                threadArgument=TestToDo.pop()
                print "Let's submit job %s on core %s"%(threadArgument,cpu)
                print "Instantiating thread"
                print "Testing the keyword arguments with:"
                kwargs={'cpu':3,'perfsuitedir':"work",'userInputFile':'TTBAR_GEN,FASTSIM.root'}
                print kwargs
                threadToDo=TestThread(threadArgument,cpu,**kwargs)
                print "Starting thread %s"%threadToDo
                threadToDo.start()
                print "Appending thread %s to the list of active threads"%threadToDo
                activeThreads[cpu]=threadToDo
            #If there is no available core, pass, there will be some checking of activeThreads, a little sleep and then another check.
            else:
                pass
        #Test activeThreads:
        for cpu in activeThreads.keys():
            if activeThreads[cpu].isAlive():
                pass
            elif cpu not in Available:
                print "About to append cpu %s to Available list"%cpu
                Available.append(cpu)
        if set(Available)==set(['0','1','2','3']) and not TestToDo:
            break
        else:
            print "Sleeping and checking again..."
            time.sleep(1)
            
    #Check we broke out of the infinite loop!
    print "WHEW! We're done... all TestToDo are done..."
    print Available
    print TestToDo

    #Next: check scenarios
    #1-many more TestToDo than Available cores
    #Test 1 done successfully.
    #2-complicated Test() class that calls other functions with args
    #3-What happens on the machine with top
    #4-What if they get killed or hang?

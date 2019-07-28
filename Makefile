SYSTEM     = x86-64_linux
LIBFORMAT  = static_pic
FILENAME = driver
#------------------------------------------------------------
#
# When you adapt this makefile to compile your CPLEX programs
# please copy this makefile and set CPLEXDIR and CONCERTDIR to
# the directories where CPLEX and CONCERT are installed.
#
#------------------------------------------------------------

CPLEXDIR      = /opt/ibm/ILOG/CPLEX_Studio128/cplex
CONCERTDIR    = /opt/ibm/ILOG/CPLEX_Studio128/concert

# ---------------------------------------------------------------------
# Compiler selection 
# ---------------------------------------------------------------------

CCC = g++ -O0
CC  = gcc -O0
JAVAC = javac 

# ---------------------------------------------------------------------
# Compiler options 
# ---------------------------------------------------------------------

CCOPT = -m64 -O -fPIC -fno-strict-aliasing -fexceptions -DNDEBUG -DIL_STD
COPT  = -m64 -fPIC -fno-strict-aliasing
JOPT  = -classpath $(CPLEXDIR)/lib/cplex.jar -O

# ---------------------------------------------------------------------
# Link options and libraries
# ---------------------------------------------------------------------

CPLEXBINDIR   = $(CPLEXDIR)/bin/$(BINDIST)
CPLEXJARDIR   = $(CPLEXDIR)/lib/cplex.jar
CPLEXLIBDIR   = $(CPLEXDIR)/lib/$(SYSTEM)/$(LIBFORMAT)
CONCERTLIBDIR = $(CONCERTDIR)/lib/$(SYSTEM)/$(LIBFORMAT)
EIGENDIR = /home/yongjis/Documents/research/partition2sMIP/functions/Eigen

CCLNDIRS  = -L$(CPLEXLIBDIR) -L$(CONCERTLIBDIR)
CLNDIRS   = -L$(CPLEXLIBDIR)
CCLNFLAGS = -lconcert -lilocplex -lcplex -lm -lpthread -ldl
CLNFLAGS  = -lcplex -lm -lpthread -ldl
JAVA      = java  -d64 -Djava.library.path=$(CPLEXDIR)/bin/x86-64_linux -classpath $(CPLEXJARDIR):


CONCERTINCDIR = $(CONCERTDIR)/include
CPLEXINCDIR   = $(CPLEXDIR)/include

CFLAGS  = $(COPT)  -I$(CPLEXINCDIR)
CCFLAGS = $(CCOPT) -I$(CPLEXINCDIR) -I$(CONCERTINCDIR) 
JCFLAGS = $(JOPT)


#------------------------------------------------------------
#  make all      : to compile the examples. 
#  make execute  : to compile and execute the examples.
#------------------------------------------------------------


all_cpp: $(FILENAME)

# ------------------------------------------------------------

clean :
	/bin/rm -rf *.o *~ *.class
	/bin/rm -rf $(C_EX) $(CX_EX) $(CPP_EX)
	/bin/rm -rf *.mps *.ord *.sos *.lp *.sav *.net *.msg *.log *.clp

$(FILENAME): $(FILENAME).o Extended.o Masterproblem.o Subproblem.o Partition.o Solution.o
	$(CCC) $(CCFLAGS) $(CCLNDIRS) $(FILENAME).o Extended.o Masterproblem.o Subproblem.o Partition.o Solution.o -o $(FILENAME) $(CCLNFLAGS)
$(FILENAME).o: $(FILENAME).cpp
	$(CCC) -c $(CCFLAGS) -I$(EIGENDIR) $(FILENAME).cpp -o $(FILENAME).o
Extended.o: Extended.cpp
	$(CCC) -c $(CCFLAGS) -I$(EIGENDIR) Extended.cpp -o Extended.o
Masterproblem.o: Masterproblem.cpp
	$(CCC) -c $(CCFLAGS) -I$(EIGENDIR) Masterproblem.cpp -o Masterproblem.o
Subproblem.o: Subproblem.cpp
	$(CCC) -c $(CCFLAGS) -I$(EIGENDIR) Subproblem.cpp -o Subproblem.o
Partition.o: Partition.cpp
	$(CCC) -c $(CCFLAGS) -I$(EIGENDIR) Partition.cpp -o Partition.o
Solution.o: Solution.cpp
	$(CCC) -c $(CCFLAGS) -I$(EIGENDIR) Solution.cpp -o Solution.o


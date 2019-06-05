#ifndef EXTENDED_H
#define EXTENDED_H

#include "adaptiveSamples.h"
#include <stdio.h>
#include <stdlib.h>
//#include <ilcplex/ilocplex.h>
//#include <ilconcert/iloenv.h>

class Extended
{
	
  private:
    IloEnv env;  //Is & needed or not?
    TSLP prob;
    STAT stat;
    IloTimer clock;

  public:
	//Constructor
    Extended(IloEnv& extend_env, const TSLP& extend_prob, STAT& extend_stat, IloTimer& extend_clock);
    
	//Deconstructor
	~Extended();

    void solve_extended();
};

#endif
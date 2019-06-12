#ifndef EXTENDED_H
#define EXTENDED_H

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "adaptiveSamples.h"

using namespace std;

class Extended
{
public:
	//Constructor
	Extended();

	//Deconstructor
	~Extended();

	void solve_extended(IloEnv& env, const TSLP& prob, STAT& stat, IloTimer& clock);

};
#endif
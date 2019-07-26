#ifndef EXTENDED_H
#define EXTENDED_H

#include "Subproblem.h"
#include "structs.h"


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
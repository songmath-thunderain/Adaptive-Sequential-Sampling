#pragma once
#ifndef SUBPROBLEM_H
#define SUBPROBLEM_H

#include "structs.h"
//#include "adaptiveSamples.h"


using namespace std;

struct Subprob
{
	IloModel suboptmodel;
	IloRangeArray suboptcon;
	IloNumVarArray subopty;
	IloCplex suboptcplex;
	IloModel subfeasmodel;
	IloRangeArray subfeascon;
	IloNumVarArray subfeasy;
	IloCplex subfeascplex;
};

class Subproblem {

public:
	Subproblem();

	~Subproblem();

	double subprob(Subprob& subp, const TSLP& prob, const IloNumArray& xvals, IloNumArray& duals, int k, bool& feasflag);

	void construct_second_opt(class IloEnv&, const TSLP&, Subprob&);

	void construct_second_feas(class IloEnv&, const TSLP&, Subprob&);

};



#endif
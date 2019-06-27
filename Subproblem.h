#pragma once
#ifndef SUBPROBLEM_H
#define SUBPROBLEM_H

#include "structs.h"


using namespace std;

class Subproblem {
private:
	IloModel suboptmodel;
	IloRangeArray suboptcon;
	IloNumVarArray subopty;
	IloCplex suboptcplex;
	IloModel subfeasmodel;
	IloRangeArray subfeascon;
	IloNumVarArray subfeasy;
	IloCplex subfeascplex;

public:
	Subproblem(IloEnv& env, const TSLP& prob);

	~Subproblem();

	double solve(const TSLP& prob, const IloNumArray& xvals, IloNumArray& duals, int k, bool& feasflag);

	void construct_second_opt(class IloEnv&, const TSLP&);

	void construct_second_feas(class IloEnv&, const TSLP&);

};



#endif
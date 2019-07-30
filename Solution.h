#pragma once
/*- mode: C++;
  Dec. 20, 2017
 */

#ifndef SOLUTION_H
#define SOLUTION_H

#include <ilcplex/ilocplex.h>
#include <vector>
#include <set>
#include <Eigen>
#include <iostream>
#include <algorithm>
#include <cmath>
extern "C" {
#include <stdio.h>
#include <stdlib.h>
}
#include <time.h>

#include "Masterproblem.h"
#include "Subproblem.h"
#include "Partition.h"
#include "structs.h"

//Move to Solution.cpp
//double t_quant = 1.282; // For simplicity, just use z quantile (independent of degree of freedom), since sample sizes are usually large in the end

using namespace std;

class Solution {
public:
	Solution();

	~Solution();

	void preprocessing(IloEnv& env, TSLP& prob);

	double solve_mean_value_model(const TSLP& prob, IloEnv& meanenv, IloNumArray& meanxvals, const vector<int>& samples);

	void computeSamplingError(double& samplingError, const vector<double>& scenObjs);

	void setup_bundle_QP(IloEnv& env, const TSLP& prob, IloCplex& cplex, IloModel& model, IloNumVarArray& x, const IloNumArray& stab_center, IloObjective& QPobj, IloRangeArray& cuts, IloRangeArray& center_cons);

	void solve_singlecut(IloEnv& env, TSLP& prob, STAT& stat, IloTimer& clock, const vector<int>& samples, VectorXf& xiterateXf);

	void solve_level(IloEnv& env, TSLP& prob, STAT& stat, IloTimer& clock, const vector<int>& samples, VectorXf& xiterateXf, int option);

	bool addToCollection(const VectorXf& dualvec, vector<DualInfo>& dualInfoCollection);

	void externalEval(IloEnv& env, Subprob& subp, const TSLP& prob, const IloNumArray& xvals, int sampleSize, vector<double>& objValVec);

	void solve_adaptive(IloEnv& env, TSLP& prob, STAT& stat, IloTimer& clock, int option, bool saaError);

	void solve_adaptive_partition(IloEnv& env, TSLP& prob, STAT& stat, IloTimer& clock, int option);

	void addInitialCuts(IloEnv& env, TSLP& prob, IloModel& mastermodel, const IloNumVarArray& x, const IloNumVarArray& theta, IloCplex& mastercplex, IloRangeArray& cutcon, const vector<int>& samplesForSol, const vector<DualInfo>& dualInfoCollection, const VectorXf& xiterateXf, const vector<VectorXf>& rhsvecs);

	bool solve_partly_inexact_bundle(IloEnv& env, TSLP& prob, STAT& stat, IloTimer& clock, const vector<int>& samples, VectorXf& xiterateXf, int option, bool initial, vector<DualInfo>& dualInfoCollection, const vector<VectorXf>& rhsvecs, int& nearOptimal, double remaintime);

	void SRP(IloEnv& env, TSLP& prob, IloTimer& clock, int nbIterEvalScens, const vector<int>& samplesForEval, double& G, double& S, const IloNumArray& xvals, const IloNumArray& xvals2, vector<double>& scenObjEval);

	void BMschedule(Sequence& seq);

	void sequentialSetup(Sequence& seq, int option);

	void finalEval(IloEnv& env, TSLP& prob, const VectorXf& xiterateXf, STAT& stat);
};

#endif

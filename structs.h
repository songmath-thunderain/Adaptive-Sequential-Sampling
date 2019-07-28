#pragma once
#ifndef STRUCTS_H
#define STRUCTS_H

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

using namespace std;

using Eigen::MatrixXf;
using Eigen::VectorXf;

typedef IloArray<IloNumArray> NumMatrix;
typedef IloArray<IloIntArray> IntMatrix;
struct TSLP
{
	int nbRows;
	int nbFirstVars;
	int nbSecVars;
	int nbScens;
	IntMatrix firstconstrind;
	NumMatrix firstconstrcoef;
	IntMatrix secondconstrind;
	NumMatrix secondconstrcoef;
	IloNumArray firstconstrlb;
	IloNumArray firstconstrub;
	IloNumArray secondconstrlb;
	IloNumArray secondconstrub;
	IloNumArray secondconstrbd;
	IloNumArray varlb;
	IloNumArray varub;
	IloNumArray firstvarlb;
	IloNumArray firstvarub;
	IloNumArray secondvarlb;
	IloNumArray secondvarub;
	IloNumArray objcoef;
	int nbFirstRows;
	int nbSecRows;
	IloIntArray nbPerRow;
	NumMatrix CoefMat;
	IntMatrix CoefInd;
	int master_solver;
	double distinct_par;
	IntMatrix addfirstconstrind;
	NumMatrix addfirstconstrcoef;
	IloNumArray addfirstconstrrhs;
	IloIntArray secondconstrsense;
	int randomseed;
	MatrixXf CoefMatXf;
	double kappa;
	double kappaf;
	double gamma;
	double increaseRate;
	double eps;
};

struct Sequence
{
	int nMax;
	double p, h, h2, eps, eps2;
	vector<int> sampleSizes;
};

struct Model
{
	IloNumVarArray y;
	IloNumArray lambda;
	IloNumVarArray x;
	IloNumVarArray eta;
};

struct STAT
{
	double solvetime;
	double mastertime;
	double qptime;
	double subtime;
	double evaltime;
	double warmstarttime;
	double warmstartcuttime;
	double refinetime;
	double relaxobjval;
	double feasobjval;
	double objval;
	int num_feas_cuts;
	int num_opt_cuts;
	int iter;
	int mainIter;
	double finalSolExactObj;
	double gapThreshold;
	int finalSampleSize;
	int nbSecLPSolves;
	int partitionsize;
	int finalpartitionsize;
};

//move to subproblem
/*
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

*/

struct DualInfo
{
	VectorXf dualvec;
	VectorXf coefvec;
	double rhs;
};

//move to Partition
struct Component
{
	vector<int> indices;
};

class IndexVal {
public:
	int ind;
	double val;
	IndexVal() {}
	IndexVal(const int& i, const double& v)
	{
		ind = i; val = v;
	}
};

static bool operator<(const IndexVal& a, const IndexVal& b) {
	if (a.val > b.val)
		return true;
	else if (a.val < b.val)
		return false;
	else if (a.ind > b.ind)
		return true;
	else
		return false;
}

static bool operator>(const IndexVal& a, const IndexVal& b) {
	if (a.val < b.val)
		return true;
	else if (a.val > b.val)
		return false;
	else if (a.ind < b.ind)
		return true;
	else
		return false;
}

#endif

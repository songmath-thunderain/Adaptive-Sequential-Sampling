/*
   Header file for master problem class.
*/
#ifndef MASTERPROBLEM_H
#define MASTERPROBLEM_H

#include <ilcplex/ilocplex.h>
#include "structs.h"
//#include "adaptiveSamples.h"
using Eigen::VectorXf;

class Masterproblem {
private:
	IloEnv env;
	TSLP prob;
	STAT stat;
	//IloTimer clock;
	vector<int> samples;
	IloModel model;
	IloNumVarArray x;
	IloNumVar theta;
	IloCplex cplex;
	// Quadratic MP
	IloEnv lenv;
	IloModel levelmodel;
	IloExpr lsum;
	IloRange rangeub;
	IloObjective lobj;
	IloNumVarArray lx;
	IloNumVar ltheta;
	IloCplex levelcplex;

public:
	Masterproblem();
	Masterproblem(IloEnv& env, TSLP& prob, STAT& stat, IloTimer& clock, const vector<int>& samples);
	Masterproblem(IloEnv& env, TSLP& prob, STAT& stat, IloTimer& clock, const vector<int>& samples, IloEnv lenv);
	~Masterproblem();
	void define_lp_model();
	void define_qp_model();
	void setup_bundle_QP(const IloNumArray& stab_center, IloObjective& QPobj, IloRangeArray& cuts, IloRangeArray& center_cons);
	void addInitialCuts(IloEnv& env, TSLP& prob, IloNumVarArray thetaArr, IloRangeArray& cutcon, const vector<int>& samplesForSol, const vector<DualInfo>& dualInfoCollection, const VectorXf& xiterateXf, const vector<VectorXf>& rhsvecs);
	IloCplex& getCplex();
	IloNumVarArray& getX();
	IloNumVar& getTheta();
	IloModel& getModel();
	IloNumVar& getLtheta();
	IloNumVarArray& getLx();
	IloModel& getLevelmodel();
	IloCplex& getLevelcplex();
	IloRange& getRangeub();
	IloObjective& getLobj();

	// Allow partition class to access private data.
	friend class Partition;
};

#endif

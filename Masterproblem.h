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
	IloNumVarArray theta_multi;
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
	void define_lp_model(int option);
	void define_qp_model();
	void setup_bundle_QP(const IloNumArray& stab_center, IloObjective& QPobj, IloRangeArray& cuts, IloRangeArray& center_cons);
	void addInitialCuts(IloEnv& env, TSLP& prob, IloRangeArray& cutcon, const vector<int>& samplesForSol, const vector<DualInfo>& dualInfoCollection, const VectorXf& xiterateXf, const vector<VectorXf>& rhsvecs);
	void add_feas_cuts(IloEnv& env, TSLP& prob, const IloNumArray& xvals, double subobjval, const VectorXf& dualvec, const Component& compo);
	IloCplex& getCplex();
	IloNumVarArray& getX();
	IloNumVar& getTheta();
	IloNumVarArray& getThetaMulti();
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

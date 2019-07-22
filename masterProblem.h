/*
   Header file for master problem class.
*/
#ifndef MASTERPROBLEM_H
#define MASTERPROBLEM_H

#include "structs.h"
#include "adaptiveSamples.h"
using Eigen::VectorXf;

class MasterProblem {
  private:
	IloEnv env;
    TSLP prob;
    STAT stat;
    //IloTimer clock;
    vector<int> samples;
    VectorXf xiterateXf;
	IloModel model;
	IloNumVarArray x;
	IloNumVar theta;
	IloCplex cplex;
    // Mean data used in solve_level
    IloEnv meanenv;
    IloNumArray meanxvals;
    double meanobj;
	vector<double> xiterate;
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
	MasterProblem();
	MasterProblem(IloEnv& env, TSLP& prob, STAT& stat, IloTimer& clock, const vector<int>& samples, VectorXf& xiterateXf);
	//MasterProblem(IloEnv& env, TSLP& prob, STAT& stat, IloTimer& clock, const vector<int>& samples, VectorXf& xiterateXf, IloEnv meanenv);
	MasterProblem(IloEnv& env, TSLP& prob, STAT& stat, IloTimer& clock, const vector<int>& samples, VectorXf& xiterateXf, IloEnv meanenv, IloEnv lenv);
    ~MasterProblem();
    void define_lp_model();
    void define_qp_model();
    void setup_bundle_QP(const IloNumArray& stab_center, IloObjective& QPobj, IloRangeArray& cuts, IloRangeArray& center_cons);
    double solve_mean_value_model(const TSLP& prob, IloEnv& meanenv, IloNumArray& meanxvals, const vector<int>& samples);
	IloCplex& getCplex();
	IloNumVarArray& getX();
	IloNumVar& getTheta();
	IloModel& getModel();
	IloNumArray& getMeanxvals();
	vector<double>& getXiterate();
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

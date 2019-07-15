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
    // JUST INPUTS?
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
    IloEnv meanenv; // JUST INPUT?
    IloNumArray meanxvals;
    double meanobj;
    // Quadratic MP
    IloEnv lenv; // JUST INPUT?
    IloModel levelmodel;
    IloNumVarArray lx;
    IloNumVar ltheta;
    IloCplex levelcplex;

    // Helper Functions
    IloRange find_constraint(int i);
    void add_objective();

  public:
	MasterProblem(IloEnv& env, TSLP& prob, STAT& stat, IloTimer& clock, const vector<int>& samples, VectorXf& xiterateXf);
	MasterProblem(IloEnv& env, TSLP& prob, STAT& stat, IloTimer& clock, const vector<int>& samples, VectorXf& xiterateXf, IloEnv meanenv);
	MasterProblem(IloEnv& env, TSLP& prob, STAT& stat, IloTimer& clock, const vector<int>& samples, VectorXf& xiterateXf, IloEnv meanenv, IloEnv lenv);
    ~MasterProblem();
	void solve();
    IloCplex& getCplex();
    void define_lp_model();
    void define_qp_model();
    void setup_bundle_QP(const IloNumArray& stab_center, IloObjective& QPobj, IloRangeArray& cuts, IloRangeArray& center_cons);
    double solve_mean_value_model(const TSLP& prob, IloEnv& meanenv, IloNumArray& meanxvals, const vector<int>& samples);

    // Allow partition to access private data.
    friend class Partition;
};

#endif

/*
   Header file for master problem class.
*/
#ifndef MASTERPROBLEM_H
#define MASTERPROBLEM_H

#include "structs.h"
//#inclue "adaptiveSamples.h"
using Eigen::VectorXf;

class MasterProblem {
  private:
    IloEnv& env;
    TSLP& prob;
    STAT& stat;
    IloTimer& clock;
    const vector<int>& samples;
    VectorXf& xiterateXf;
    // *** only in solve_level
      // Setup
      IloEnv meanenv;
      IloNumArray meanxvals;
      double meanobj;
      // quadratic
      IloEnv lenv;
      IloModel levelmodel;
      IloNumVarArray lx;
      IloNumVar ltheta;
      IloCplex levelcplex;

    // ***
    IloModel model;
  	IloNumVarArray x;
  	IloNumVar theta;
    IloCplex cplex;

    // Helper Functions
    IloRange find_constraint(int i);
    void add_objective();


  public:
    MasterProblem();
    ~MasterProblem();
    void getCplex();
    void define_lp_model();
    void define_qp_model();
    void setup_bundle_QP(const IloNumArray& stab_center, IloObjective& QPobj, IloRangeArray& cuts, IloRangeArray& center_cons);
    double solve_mean_value_model(const TSLP& prob, IloEnv& meanenv, IloNumArray& meanxvals, const vector<int>& samples);

    // Allow partition to access private data.
    friend class Partition;
};

#endif

/*
   Header file for master problem class.
*/
#ifndef MASTERPROBLEM_H
#define MASTERPROBLEM_H

#include "structs.h"
//#inclue "adaptiveSamples.h"
using Eigen::VectorXf;

class masterProblem {
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
    IloModel mastermodel;
  	IloNumVarArray x;
  	IloNumVar theta;
    IloCplex mastercplex;

    // Helper Functions
    IloRange find_constraint(int i);
    void add_objective();

  public:
    masterProblem();
    ~masterProblem();
    void first_stage_constraints();
    void define_first_level_constraint();
};

#endif

/*
   Header file for master problem class.
*/
#ifndef MASTERPROBLEM_H
#define MASTERPROBLEM_H

using Eigen::VectorXf;

void solve_singlecut(IloEnv& env, TSLP& prob, STAT& stat, IloTimer& clock, const vector<int>& samples, VectorXf& xiterateXf);
void solve_level(IloEnv& env, TSLP& prob, STAT& stat, IloTimer& clock, const vector<int>& samples, VectorXf& xiterateXf, int option);

class masterProblem {
  private:
    IloEnv& env;
    TSLP& prob;
    STAT& stat;
    IloTimer& clock;
    const vector<int>& samples;
    VectorXf& xiterateXf;
    // *** only in solve_level
    IloEnv meanenv;
    IloNumArray meanxvals;
    double meanobj;
    // ***
    IloModel mastermodel;
  	IloNumVarArray x;
  	IloNumVar theta;
    IloCplex mastercplex;

    // Helper Functions
    IloRange find_constraint(int i);
    void add_objective();
    // ***

  public:
    masterProblem();
    ~masterProblem();
    void first_stage_constraints();
};

#endif

/*
  Master problem of optimization.
  Interacts with subproblems to improve approximations.
*/

#include "masterProblem.h"

  /*
    Constructor
  */
  masterProblem::masterProblem(/* input variables at least need option */) {
    // *** only solve_level
    meanxvals(meanenv);
    meanobj = solve_mean_value_model(TSLP& prob, IloEnv meanenv, IloNumArray meanxvals, const vector<int>& samples);
    // ***
    mastermodel(env);
    x(env, prob.firstvarlb, prob.firstvarub);
    theta(env, 0, IloInfinity);
    mastercplex(mastermodel);
  }

  /*
    Destructor
  */
  masterProblem::~masterProblem() {
    mastercplex.end();
    mastermodel.end();
    x.end();
  }

  /*
    Set initial constraints for optimization problem.
  */
  void first_stage_constraints() {
    for (int i = 0; i < prob.firstconstrind.getSize(); ++i)
  	{
      mastermodel.add(find_constraint(i));
  	}
  }

  /*
    Helper function for setting initial constraints.
  */
  IloRange find_constraint(int i) {
    IloExpr lhs(env);
    for (int j = 0; j < prob.firstconstrind[i].getSize(); ++j)
      lhs += x[prob.firstconstrind[i][j]]*prob.firstconstrcoef[i][j];
    IloRange range(env, prob.firstconstrlb[i], lhs, prob.firstconstrub[i]);
    lhs.end();
    return range;
  }

  /*

  */
  void add_objective() {
    // Adding objective
  	IloExpr obj(env);
  	for (int i = 0; i < prob.nbFirstVars; ++i)
  		obj += x[i]*prob.objcoef[i];
  	// single cut
  	obj += theta;
  	mastermodel.add(IloMinimize(env, obj));
  	obj.end();
  	IloCplex mastercplex(mastermodel);
  	mastercplex.setParam(IloCplex::TiLim,10800);
  	mastercplex.setParam(IloCplex::Threads, 1);
  	// Barrier
  	mastercplex.setParam(IloCplex::RootAlg, 2);
  	mastercplex.setParam(IloCplex::BarDisplay, 0);
  	mastercplex.setParam(IloCplex::SimDisplay, 0);
  	mastercplex.setOut(env.getNullStream());
  	//mastercplex.setParam(IloCplex::EpOpt, 1e-4);
  }
